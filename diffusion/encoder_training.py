from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusion.data_pipeline import RandomRotationTranslation3D, create_dataloader
from diffusion.encoder_architecture import PackedSequenceEncoder
from diffusion.encoder_losses import DINOv2Loss


class EMA:
    """
    Exponential Moving Average for teacher model in DINO.
    """

    def __init__(self, model: nn.Module, decay: float = 0.996):
        """
        Args:
            model: Student model to track with EMA
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class ProteinEncoderTrainer:
    """
    Trainer for protein structure encoder using DINO v2 self-supervised learning.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        teacher_momentum: float = 0.996,
        mask_ratio: float = 0.15,
        use_wandb: bool = True,
    ):
        """
        Args:
            student: Student encoder model
            teacher: Teacher encoder model (EMA of student)
            criterion: Loss function (e.g., DINOv2Loss)
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            teacher_momentum: EMA momentum for teacher
            mask_ratio: Ratio of residues to mask for iBOT loss
            use_wandb: Whether to log to wandb
        """
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mask_ratio = mask_ratio
        self.use_wandb = use_wandb

        # Teacher is not trainable
        for param in self.teacher.parameters():
            param.requires_grad = False

        # EMA for teacher
        self.teacher_ema = EMA(self.student, decay=teacher_momentum)

        self.global_step = 0
        self.epoch = 0
        self.save_dir = None

    def train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data

        Returns:
            Dictionary of average losses for the epoch
        """
        self.student.train()
        self.teacher.eval()

        epoch_losses = {"total": 0.0, "dino": 0.0, "koleo": 0.0, "ibot": 0.0}
        num_batches = 0

        for _batch_idx, batch in enumerate(dataloader):
            # Skip batches where all samples were filtered out
            if batch is None:
                continue

            # Get protein structures
            structures = batch["coords"]  # List of [n_residues, n_atoms, 3]

            # Forward pass through student
            student_out_dict = self.student(structures, return_projected=True)
            # student_backbone = student_out_dict["backbone"]  # List of [n_residues, embed_dim]
            student_projected = student_out_dict["projected"]  # List of [n_residues, proj_dim]

            # Forward pass through teacher (no grad)
            with torch.no_grad():
                teacher_out_dict = self.teacher(structures, return_projected=True)
                # teacher_backbone = teacher_out_dict["backbone"]
                teacher_projected = teacher_out_dict["projected"]

            # Prepare inputs for DINOv2Loss
            # For DINO loss: use global average pooling on PROJECTED embeddings as [CLS] tokens
            # For KoLeo loss: will be applied to these same projected [CLS] tokens inside the loss
            student_cls = [out.mean(dim=0, keepdim=True) for out in student_projected]
            teacher_cls = [out.mean(dim=0, keepdim=True) for out in teacher_projected]

            # For iBOT loss: use PROJECTED residue embeddings as patch tokens
            # Process all structures in the batch by padding them to same length
            if len(student_projected) > 0:
                # Stack all structures: each becomes a separate batch item
                # Result: [B, N, D] where B=batch_size, N=max_residues (padded), D=proj_dim
                max_residues = max(s.shape[0] for s in student_projected)
                batch_size = len(student_projected)
                proj_dim = student_projected[0].shape[1]

                # Initialize padded tensors
                student_patch_tokens = torch.zeros(
                    batch_size,
                    max_residues,
                    proj_dim,
                    device=student_projected[0].device,
                    dtype=student_projected[0].dtype,
                )
                teacher_patch_tokens = torch.zeros(
                    batch_size,
                    max_residues,
                    proj_dim,
                    device=teacher_projected[0].device,
                    dtype=teacher_projected[0].dtype,
                )
                student_masks = torch.zeros(
                    batch_size, max_residues, device=student_projected[0].device, dtype=torch.bool
                )

                # Fill in the data and create masks for each structure
                for i, (s_proj, t_proj) in enumerate(
                    zip(student_projected, teacher_projected, strict=True)
                ):
                    n_res = s_proj.shape[0]
                    student_patch_tokens[i, :n_res] = s_proj
                    teacher_patch_tokens[i, :n_res] = t_proj

                    # Create random mask for iBOT (mask residues for prediction)
                    # Only mask actual residues, not padding
                    mask = torch.rand(n_res, device=s_proj.device) < self.mask_ratio
                    student_masks[i, :n_res] = mask
            else:
                student_patch_tokens = None
                teacher_patch_tokens = None
                student_masks = None

            losses = self.criterion(
                student_output=student_cls,
                teacher_output=teacher_cls,
                epoch=self.epoch,
                student_patch_tokens=student_patch_tokens,
                teacher_patch_tokens=teacher_patch_tokens,
                student_masks_flat=student_masks,
            )

            loss = losses["total"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm before clipping for logging
            total_norm = 0.0
            for p in self.student.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            # Gradient clipping (similar to AlphaFold/transformers)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)

            # print(f"Gradient norm: {total_norm:.2g}, Loss: {loss.item():.4f}")

            self.optimizer.step()

            # Update teacher with EMA
            self.teacher_ema.update()
            self._copy_params_to_teacher()

            # Logging
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1

            # Log to wandb and print every 10 batches
            if self.global_step % 10 == 0:
                log_dict = {f"train/{key}": value.item() for key, value in losses.items()}
                log_dict["train/lr"] = self.optimizer.param_groups[0]["lr"]
                log_dict["train/gradient_norm"] = total_norm
                log_dict["train/step"] = self.global_step

                if self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)

                # Print losses
                loss_str = ", ".join(
                    [f"{key}: {value.item():.4f}" for key, value in losses.items()]
                )
                print(
                    f"Step {self.global_step}: {loss_str}, grad_norm: {total_norm:.2g},",
                    f" lr: {self.optimizer.param_groups[0]['lr']:.2e}",
                )

            self.global_step += 1

            # Save checkpoint every 10 batches
            if self.global_step % 10 == 0 and hasattr(self, "save_dir") and self.save_dir:
                checkpoint_path = self.save_dir / f"checkpoint_step_{self.global_step}.pt"
                self.save_checkpoint(checkpoint_path)
                # print(f"Saved checkpoint at step {self.global_step}")

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def _copy_params_to_teacher(self):
        """Copy EMA parameters from student to teacher."""
        for (name_s, _param_s), (_name_t, param_t) in zip(
            self.student.named_parameters(), self.teacher.named_parameters(), strict=True
        ):
            param_t.data.copy_(self.teacher_ema.shadow[name_s])

    def train(self, dataloader, num_epochs: int, save_dir: str | None = None):
        """
        Main training loop.

        Args:
            dataloader: Training dataloader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

        print("\nTraining complete!")

        # Save final model
        if self.save_dir:
            self.save_checkpoint(self.save_dir / "final_model.pt")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "teacher_ema_shadow": self.teacher_ema.shadow,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        # print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.teacher.load_state_dict(checkpoint["teacher_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.teacher_ema.shadow = checkpoint["teacher_ema_shadow"]
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {path}")


def create_model(
    n_atoms: int = 37,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    proj_output_dim: int = 65536,
    proj_hidden_dim: int = 2048,
    proj_bottleneck_dim: int = 256,
) -> PackedSequenceEncoder:
    """Create a protein structure encoder model with projection head."""
    return PackedSequenceEncoder(
        n_atoms=n_atoms,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        proj_output_dim=proj_output_dim,
        proj_hidden_dim=proj_hidden_dim,
        proj_bottleneck_dim=proj_bottleneck_dim,
    )


def main():
    """Main training function."""
    # Configuration (DINOv2 paper defaults)
    config = {
        # Model architecture
        "n_atoms": 37,
        "embed_dim": 768 // 4,
        "depth": 12,
        "num_heads": 12 // 2,
        "mlp_ratio": 4.0,
        # Projection head parameters
        "proj_output_dim": 65536 // 16,  # Projection space dimension (DINOv2 default)
        "proj_hidden_dim": 2048 // 16,  # Projection head hidden dimension
        "proj_bottleneck_dim": 256 // 16,  # Projection head bottleneck dimension
        # Training hyperparameters
        "batch_size": 32 // 2,  # 4,
        "num_epochs": 100,
        "lr": 1e-4 / 10000,
        "weight_decay": 0.05,
        # DINOv2 loss parameters (from paper)
        "teacher_momentum": 0.996,  # EMA momentum for teacher
        "warmup_teacher_temp": 0.04,  # Initial teacher temperature
        "teacher_temp": 0.07,  # Final teacher temperature
        "warmup_teacher_temp_epochs": 30,  # Warmup epochs
        "student_temp": 0.1,  # Student temperature (fixed)
        "lambda_koleo": 0.1,  # KoLeo loss weight (paper default)
        "lambda_ibot": 10.0,  # 1.0,  # iBOT loss weight (paper default)
        "mask_ratio": 0.15,  # Residue masking ratio for iBOT (BERT/iBOT default)
        # Data
        "pdb_dir": "psp-lite/",  # 'CASP14/', #'psp-lite/',
        "file_pattern": "*.pdb",
        # Logging
        "save_dir": "checkpoints",
        "use_wandb": False,
    }

    # Initialize wandb
    if config["use_wandb"]:
        wandb.init(
            project="protein-structure-encoder",
            config=config,
        )

    # Create models
    print("Creating models...")
    student = create_model(
        n_atoms=config["n_atoms"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        proj_output_dim=config["proj_output_dim"],
        proj_hidden_dim=config["proj_hidden_dim"],
        proj_bottleneck_dim=config["proj_bottleneck_dim"],
    )

    teacher = create_model(
        n_atoms=config["n_atoms"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        proj_output_dim=config["proj_output_dim"],
        proj_hidden_dim=config["proj_hidden_dim"],
        proj_bottleneck_dim=config["proj_bottleneck_dim"],
    )

    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())

    # Create loss function with DINOv2 paper defaults
    print("Creating loss function...")
    criterion = DINOv2Loss(
        out_dim=config["proj_output_dim"],  # Use projection dimension for DINO loss
        warmup_teacher_temp=config["warmup_teacher_temp"],
        teacher_temp=config["teacher_temp"],
        warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
        student_temp=config["student_temp"],
        lambda_koleo=config["lambda_koleo"],
        lambda_ibot=config["lambda_ibot"],
        patch_out_dim=config["proj_output_dim"],  # Use projection dimension for iBOT loss
    )

    # Setup teacher temperature schedule
    criterion.setup_teacher_temp_schedule(config["num_epochs"])

    # Create optimizer and scheduler
    print("Creating optimizer...")
    optimizer = AdamW(
        student.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"],
        eta_min=config["lr"] * 0.01,
    )

    # Create dataloader
    print("Creating dataloader...")
    transform = RandomRotationTranslation3D(translation_range=5.0)
    dataloader = create_dataloader(
        pdb_dir=config["pdb_dir"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        transform=transform,
        file_pattern=config["file_pattern"],
    )

    # Print model info
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print("\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: {total_params * 4 / 1024**2:.2f} MB (fp32)")

    # Create trainer
    print("\nCreating trainer...")
    trainer = ProteinEncoderTrainer(
        student=student,
        teacher=teacher,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        teacher_momentum=config["teacher_momentum"],
        mask_ratio=config["mask_ratio"],
        use_wandb=config["use_wandb"],
    )

    # Train
    print("Starting training...")
    trainer.train(
        dataloader=dataloader,
        num_epochs=config["num_epochs"],
        save_dir=config["save_dir"],
    )

    if config["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
