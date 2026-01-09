import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class DINOLoss(nn.Module):
    """
    DINO v2 loss function with centering and sharpening.

    Computes cross-entropy between student and teacher outputs,
    with teacher sharpening (lower temperature) and centering to prevent collapse.
    """

    def __init__(
        self,
        out_dim: int,
        n_crops: int = 2,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        """
        Args:
            out_dim: Dimension of the output (number of prototypes/classes)
            n_crops: Number of global crops (typically 2)
            warmup_teacher_temp: Initial teacher temperature
            teacher_temp: Final teacher temperature
            warmup_teacher_temp_epochs: Epochs to warm up teacher temperature
            student_temp: Student temperature (higher = softer)
            center_momentum: EMA momentum for centering
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Teacher temperature schedule
        self.teacher_temp_schedule = None
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs

    def setup_teacher_temp_schedule(self, n_epochs: int):
        """Setup linear warmup schedule for teacher temperature."""
        self.teacher_temp_schedule = torch.linspace(
            self.warmup_teacher_temp,
            self.teacher_temp,
            self.warmup_teacher_temp_epochs
        )
        # Extend with constant temperature after warmup
        self.teacher_temp_schedule = torch.cat([
            self.teacher_temp_schedule,
            torch.full((n_epochs - self.warmup_teacher_temp_epochs,), self.teacher_temp)
        ])

    def forward(self, student_output, teacher_output, epoch):
        """
        Args:
            student_output: List of tensors from student network for all crops
                           Each tensor: [batch_size, out_dim]
            teacher_output: List of tensors from teacher network for global crops only
                           Each tensor: [batch_size, out_dim]
            epoch: Current epoch for temperature scheduling

        Returns:
            loss: Scalar loss value
        """
        # Get teacher temperature
        if self.teacher_temp_schedule is not None and epoch < len(self.teacher_temp_schedule):
            teacher_temp = self.teacher_temp_schedule[epoch]
        else:
            teacher_temp = self.teacher_temp

        # Compute teacher outputs (only for global crops)
        teacher_out = torch.cat([t for t in teacher_output], dim=0)
        teacher_out = teacher_out.detach()  # Stop gradient

        # Center and sharpen teacher outputs
        teacher_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)

        # Compute student outputs for all crops
        student_out = torch.cat([s for s in student_output], dim=0)
        student_out = F.log_softmax(student_out / self.student_temp, dim=-1)

        # Split student outputs by crop
        n_global_crops = len(teacher_output)
        n_student_chunks = len(student_output)
        student_chunks = student_out.chunk(n_student_chunks)
        teacher_chunks = teacher_out.chunk(n_global_crops)

        # Cross-entropy between all student crops and all teacher crops
        total_loss = 0
        n_loss_terms = 0

        for t_idx, t_crop in enumerate(teacher_chunks):
            for s_idx, s_crop in enumerate(student_chunks):
                # Don't compare a crop with itself
                if t_idx == s_idx and s_idx < n_global_crops:
                    continue

                # Cross-entropy: -sum(teacher * log(student))
                loss = -torch.sum(t_crop * s_crop, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center with EMA
        self.update_center(teacher_out)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output centering with EMA."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # Distributed: average across all GPUs
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class KoLeoLoss(nn.Module):
    """
    KoLeo regularizer from DINO v2.

    Encourages uniformity in the feature space by penalizing
    features that are too close to their nearest neighbors.
    This helps prevent dimensional collapse.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features tensor [batch_size, feature_dim]

        Returns:
            loss: Scalar KoLeo loss
        """
        # Normalize features
        x = F.normalize(x, dim=-1, p=2)

        # Compute pairwise distances
        # [batch_size, batch_size]
        pdist = torch.cdist(x, x, p=2)

        # For each sample, find distance to nearest neighbor (excluding itself)
        # Set diagonal to inf so we don't select the point itself
        pdist = pdist + torch.eye(pdist.size(0), device=pdist.device) * 1e6
        min_distances = torch.min(pdist, dim=-1)[0]

        # KoLeo loss: negative log of nearest neighbor distances
        # We want to maximize distances, so minimize negative log
        loss = -torch.log(min_distances + self.eps).mean()

        return loss


class iBOTLoss(nn.Module):
    """
    iBOT (Image BERT) loss for masked image modeling in DINO v2.

    Predicts masked patches using the teacher network as targets.
    Similar to BERT masking but for images.
    """

    def __init__(
        self,
        out_dim: int,
        patch_out_dim: int = None,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        """
        Args:
            out_dim: Dimension of CLS token output
            patch_out_dim: Dimension of patch token outputs (if None, same as out_dim)
            teacher_temp: Teacher temperature for sharpening
            student_temp: Student temperature
            center_momentum: EMA momentum for centering
        """
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        patch_out_dim = patch_out_dim or out_dim
        self.register_buffer("center", torch.zeros(1, patch_out_dim))

    def forward(
        self,
        student_patch_tokens,
        teacher_patch_tokens,
        student_masks=None
    ):
        """
        Args:
            student_patch_tokens: Patch tokens from student [batch_size, n_patches, dim]
            teacher_patch_tokens: Patch tokens from teacher [batch_size, n_patches, dim]
            student_masks: Boolean mask indicating which patches were masked [batch_size, n_patches]

        Returns:
            loss: Scalar loss value
        """
        # Process teacher outputs
        teacher_patch_tokens = teacher_patch_tokens.detach()
        teacher_patch_tokens = F.softmax(
            (teacher_patch_tokens - self.center) / self.teacher_temp, dim=-1
        )

        # Process student outputs
        student_patch_tokens = F.log_softmax(
            student_patch_tokens / self.student_temp, dim=-1
        )

        # If masks provided, only compute loss on masked patches
        if student_masks is not None:
            # Flatten and select masked patches
            batch_size, n_patches, dim = student_patch_tokens.shape
            student_masked = student_patch_tokens[student_masks]
            teacher_masked = teacher_patch_tokens[student_masks]

            # Cross-entropy loss
            loss = -torch.sum(teacher_masked * student_masked, dim=-1).mean()
        else:
            # Loss on all patches
            loss = -torch.sum(teacher_patch_tokens * student_patch_tokens, dim=-1).mean()

        # Update center
        self.update_center(teacher_patch_tokens)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center with EMA."""
        # Average over batch and patches
        batch_center = torch.mean(teacher_output, dim=[0, 1], keepdim=True)

        # Distributed: average across all GPUs
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOv2Loss(nn.Module):
    """
    Combined DINO v2 loss with DINO, KoLeo, and optional iBOT losses.
    """

    def __init__(
        self,
        out_dim: int,
        n_crops: int = 2,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        lambda_koleo: float = 0.1,
        lambda_ibot: float = 1.0,
        patch_out_dim: int = None,
    ):
        """
        Args:
            out_dim: Dimension of the output
            n_crops: Number of global crops
            warmup_teacher_temp: Initial teacher temperature
            teacher_temp: Final teacher temperature
            warmup_teacher_temp_epochs: Epochs to warm up temperature
            student_temp: Student temperature
            center_momentum: EMA momentum for centering
            lambda_koleo: Weight for KoLeo loss
            lambda_ibot: Weight for iBOT loss
            patch_out_dim: Dimension of patch outputs for iBOT
        """
        super().__init__()

        self.lambda_koleo = lambda_koleo
        self.lambda_ibot = lambda_ibot

        # Main DINO loss
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            n_crops=n_crops,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )

        # KoLeo regularization
        if lambda_koleo > 0:
            self.koleo_loss = KoLeoLoss()

        # iBOT loss
        if lambda_ibot > 0:
            self.ibot_loss = iBOTLoss(
                out_dim=out_dim,
                patch_out_dim=patch_out_dim,
                teacher_temp=teacher_temp,
                student_temp=student_temp,
                center_momentum=center_momentum,
            )

    def forward(
        self,
        student_output,
        teacher_output,
        student_local_cls=None,
        epoch=0,
        student_patch_tokens=None,
        teacher_patch_tokens=None,
        student_masks=None,
    ):
        """
        Args:
            student_output: List of CLS token outputs from student for all crops
            teacher_output: List of CLS token outputs from teacher for global crops
            student_local_cls: Optional CLS tokens from local crops for KoLeo
            epoch: Current epoch
            student_patch_tokens: Optional patch tokens for iBOT
            teacher_patch_tokens: Optional patch tokens for iBOT
            student_masks: Optional masks for iBOT

        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}

        # Main DINO loss
        dino_loss = self.dino_loss(student_output, teacher_output, epoch)
        losses['dino'] = dino_loss
        total_loss = dino_loss

        # KoLeo regularization (on local crops if available)
        if self.lambda_koleo > 0 and student_local_cls is not None:
            koleo_loss = self.koleo_loss(student_local_cls)
            losses['koleo'] = koleo_loss
            total_loss = total_loss + self.lambda_koleo * koleo_loss

        # iBOT loss
        if self.lambda_ibot > 0 and student_patch_tokens is not None and teacher_patch_tokens is not None:
            ibot_loss = self.ibot_loss(student_patch_tokens, teacher_patch_tokens, student_masks)
            losses['ibot'] = ibot_loss
            total_loss = total_loss + self.lambda_ibot * ibot_loss

        losses['total'] = total_loss
        return losses
