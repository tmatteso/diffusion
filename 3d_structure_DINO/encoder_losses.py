import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)
        self.apply_center_update()

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3
    ):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        B = n_masked_patches_tensor
        dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(
            dim=-1
        ).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)
        self.apply_center_update()

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class GramAnchoringLoss(nn.Module):
    """
    Gram Anchoring loss from DINOv3 for stabilizing dense feature learning.

    During extended training, dense features can degrade even as global metrics improve.
    Gram Anchoring addresses this by constraining the student's Gram matrix (pairwise
    patch similarities) to stay close to that of an earlier, more stable "Gram teacher".

    This allows local features to evolve freely while preserving their relative structure.

    Reference: DINOv3 (https://arxiv.org/abs/2508.10104)
    """

    def __init__(self, normalize: bool = True, loss_type: str = "mse"):
        """
        Args:
            normalize: Whether to L2-normalize patch tokens before computing Gram matrix
            loss_type: Loss type for comparing Gram matrices ('mse' or 'cosine')
        """
        super().__init__()
        self.normalize = normalize
        self.loss_type = loss_type

    def compute_gram_matrix(self, patch_tokens):
        """
        Compute the Gram matrix encoding pairwise similarities between patches.

        Args:
            patch_tokens: (B, N, D) tensor of patch embeddings

        Returns:
            gram: (B, N, N) tensor of pairwise similarities
        """
        if self.normalize:
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1, eps=1e-8)

        # Gram matrix: G_ij = <patch_i, patch_j>
        # Shape: (B, N, D) @ (B, D, N) -> (B, N, N)
        gram = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))
        return gram

    def forward(self, student_patch_tokens, gram_teacher_patch_tokens):
        """
        Compute Gram Anchoring loss between student and Gram teacher.

        Args:
            student_patch_tokens: (B, N, D) patch embeddings from student
            gram_teacher_patch_tokens: (B, N, D) patch embeddings from Gram teacher
                                       (an earlier checkpoint of the model)

        Returns:
            loss: Scalar loss value
        """
        # Compute Gram matrices
        student_gram = self.compute_gram_matrix(student_patch_tokens)
        with torch.no_grad():
            teacher_gram = self.compute_gram_matrix(gram_teacher_patch_tokens)

        if self.loss_type == "mse":
            # MSE loss between Gram matrices
            loss = F.mse_loss(student_gram, teacher_gram)
        elif self.loss_type == "cosine":
            # Flatten spatial dimensions and compute cosine similarity
            B = student_gram.shape[0]
            student_flat = student_gram.view(B, -1)
            teacher_flat = teacher_gram.view(B, -1)
            # Cosine similarity loss (1 - cosine_sim)
            cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
            loss = (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. 2018."""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(student_output)  # noqa: E741
        distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        return loss


class DINOv3Loss(nn.Module):
    """
    Combined DINO v3 loss with DINO, KoLeo, iBOT, and Gram Anchoring losses.

    According to the DINOv2 paper, the total loss is:
        L = L_dino + λ_koleo * L_koleo + λ_ibot * L_ibot

    DINOv3 adds Gram Anchoring to stabilize dense features:
        L = L_dino + λ_koleo * L_koleo + λ_ibot * L_ibot + λ_gram * L_gram

    where:
    - L_dino: cross-entropy loss between student and teacher [CLS] tokens
    - L_koleo: Kozachenko-Leonenko entropy regularizer on student outputs
    - L_ibot: masked patch prediction loss (iBOT)
    - L_gram: Gram Anchoring loss for dense feature stability (DINOv3)
    """

    def __init__(
        self,
        out_dim: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        lambda_koleo: float = 0.1,
        lambda_ibot: float = 1.0,
        lambda_gram: float = 1.0,
        enable_gram_after_epoch: int = 0,
        patch_out_dim: int = None,
    ):
        """
        Args:
            out_dim: Dimension of the [CLS] token output
            warmup_teacher_temp: Initial teacher temperature
            teacher_temp: Final teacher temperature after warmup
            warmup_teacher_temp_epochs: Number of epochs to warm up temperature
            student_temp: Student temperature (fixed)
            center_momentum: EMA momentum for centering
            lambda_koleo: Weight for KoLeo loss (paper uses 0.1)
            lambda_ibot: Weight for iBOT loss (paper uses 1.0)
            lambda_gram: Weight for Gram Anchoring loss (DINOv3)
            enable_gram_after_epoch: Epoch to enable Gram Anchoring (0 = always on)
            patch_out_dim: Dimension of patch token outputs for iBOT
        """
        super().__init__()

        self.lambda_koleo = lambda_koleo
        self.lambda_ibot = lambda_ibot
        self.lambda_gram = lambda_gram
        self.enable_gram_after_epoch = enable_gram_after_epoch

        # Teacher temperature scheduling
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule = None

        # Main DINO loss (cross-entropy on [CLS] tokens)
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            student_temp=student_temp,
            center_momentum=center_momentum,
        )

        # KoLeo regularization (entropy regularizer)
        if lambda_koleo > 0:
            self.koleo_loss = KoLeoLoss()
        else:
            self.koleo_loss = None

        # iBOT patch loss (masked patch prediction)
        if lambda_ibot > 0 and patch_out_dim is not None:
            self.ibot_patch_loss = iBOTPatchLoss(
                patch_out_dim=patch_out_dim,
                student_temp=student_temp,
                center_momentum=center_momentum,
            )
        else:
            self.ibot_patch_loss = None

        # Gram Anchoring loss (DINOv3 - dense feature stabilization)
        if lambda_gram > 0:
            self.gram_anchoring_loss = GramAnchoringLoss(normalize=True, loss_type="mse")
        else:
            self.gram_anchoring_loss = None

    def setup_teacher_temp_schedule(self, num_epochs):
        """
        Setup teacher temperature schedule with linear warmup.

        Args:
            num_epochs: Total number of training epochs
        """
        if self.warmup_teacher_temp_epochs > 0:
            # Linear warmup from warmup_teacher_temp to teacher_temp
            warmup_schedule = torch.linspace(
                self.warmup_teacher_temp, self.teacher_temp, self.warmup_teacher_temp_epochs
            )
            # Constant temperature after warmup
            constant_schedule = torch.full(
                (num_epochs - self.warmup_teacher_temp_epochs,), self.teacher_temp
            )
            self.teacher_temp_schedule = torch.cat([warmup_schedule, constant_schedule])
        else:
            self.teacher_temp_schedule = torch.full((num_epochs,), self.teacher_temp)

    def get_teacher_temp(self, epoch):
        """Get the teacher temperature for the current epoch."""
        if self.teacher_temp_schedule is not None:
            return self.teacher_temp_schedule[
                min(epoch, len(self.teacher_temp_schedule) - 1)
            ].item()
        return self.teacher_temp

    def is_gram_anchoring_enabled(self, epoch):
        """
        Check if Gram Anchoring should be active for the current epoch.

        Gram Anchoring is typically enabled after initial training to allow
        features to develop before anchoring their relative structure.

        Args:
            epoch: Current training epoch

        Returns:
            bool: True if Gram Anchoring should be applied
        """
        return (
            self.gram_anchoring_loss is not None
            and epoch >= self.enable_gram_after_epoch
        )

    def should_update_gram_teacher(self, epoch, update_interval=10):
        """
        Check if the Gram teacher should be updated at this epoch.

        In DINOv3, the Gram teacher is periodically updated to a more recent
        checkpoint while still maintaining some stability in the anchor.

        Args:
            epoch: Current training epoch
            update_interval: How often (in epochs) to update the Gram teacher

        Returns:
            bool: True if the Gram teacher should be updated this epoch
        """
        if not self.is_gram_anchoring_enabled(epoch):
            return False
        # Update at the start of Gram anchoring and at regular intervals after
        epochs_since_gram_start = epoch - self.enable_gram_after_epoch
        return epochs_since_gram_start % update_interval == 0

    def forward(
        self,
        student_output,
        teacher_output,
        epoch=0,
        student_patch_tokens=None,
        teacher_patch_tokens=None,
        student_masks_flat=None,
        gram_teacher_patch_tokens=None,
    ):
        """
        Compute the combined DINOv2 loss.

        Args:
            student_output: List of student [CLS] token outputs [B, D] for each crop
            teacher_output: List of teacher [CLS] token outputs [B, D] for global crops
            epoch: Current epoch (for teacher temperature scheduling)
            student_patch_tokens: Optional [B, N, D] patch tokens from student (for iBOT)
            teacher_patch_tokens: Optional [B, N, D] patch tokens from teacher (for iBOT)
            student_masks_flat: Optional [B, N] binary mask indicating masked patches (for iBOT)
            gram_teacher_patch_tokens: Optional [B, N, D] patch tokens from Gram teacher
                                       (an earlier checkpoint for Gram Anchoring, DINOv3)

        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}

        # Get current teacher temperature
        teacher_temp = self.get_teacher_temp(epoch)

        # 1. DINO loss: cross-entropy between student and teacher [CLS] tokens
        # Center and sharpen teacher outputs
        teacher_out_softmaxed_centered = []
        for t in teacher_output:
            t_centered = self.dino_loss.softmax_center_teacher(t, teacher_temp)
            teacher_out_softmaxed_centered.append(t_centered)

        # Compute DINO loss
        dino_loss = self.dino_loss(student_output, teacher_out_softmaxed_centered)
        losses["dino"] = dino_loss
        total_loss = dino_loss

        # Update teacher center
        with torch.no_grad():
            for t in teacher_output:
                self.dino_loss.update_center(t)

        # 2. KoLeo regularization: entropy regularizer on student outputs
        if self.koleo_loss is not None and len(student_output) > 0:
            # Apply KoLeo to all student outputs (concatenated)
            student_output_cat = torch.cat(list(student_output), dim=0)
            koleo_loss = self.koleo_loss(student_output_cat)
            losses["koleo"] = koleo_loss
            total_loss = total_loss + self.lambda_koleo * koleo_loss
        else:
            losses["koleo"] = torch.tensor(0.0, device=total_loss.device)

        # 3. iBOT loss: masked patch prediction
        if (
            self.ibot_patch_loss is not None
            and student_patch_tokens is not None
            and teacher_patch_tokens is not None
            and student_masks_flat is not None
        ):

            # Center and sharpen teacher patch tokens
            teacher_patch_softmaxed = self.ibot_patch_loss.softmax_center_teacher(
                teacher_patch_tokens, teacher_temp
            )

            # Compute iBOT loss
            ibot_loss = self.ibot_patch_loss(
                student_patch_tokens, teacher_patch_softmaxed, student_masks_flat
            )
            losses["ibot"] = ibot_loss
            total_loss = total_loss + self.lambda_ibot * ibot_loss

            # Update teacher patch center
            with torch.no_grad():
                self.ibot_patch_loss.update_center(teacher_patch_tokens)
        else:
            losses["ibot"] = torch.tensor(0.0, device=total_loss.device)

        # 4. Gram Anchoring loss: stabilize dense features (DINOv3)
        # Only enabled after enable_gram_after_epoch to allow initial feature learning
        gram_enabled = (
            self.gram_anchoring_loss is not None
            and epoch >= self.enable_gram_after_epoch
            and student_patch_tokens is not None
            and gram_teacher_patch_tokens is not None
        )
        if gram_enabled:
            gram_loss = self.gram_anchoring_loss(
                student_patch_tokens, gram_teacher_patch_tokens
            )
            losses["gram"] = gram_loss
            total_loss = total_loss + self.lambda_gram * gram_loss
        else:
            losses["gram"] = torch.tensor(0.0, device=total_loss.device)

        losses["total"] = total_loss
        return losses
