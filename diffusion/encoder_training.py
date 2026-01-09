import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from diffusion.encoder_architecture import PackedSequenceEncoder, ResidueEmbedding, StructureTransformerEncoder
from diffusion.data_pipeline import create_dataloader, CenterProtein, RandomRotationTranslation3D
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
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        teacher_momentum: float = 0.996,
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
            use_wandb: Whether to log to wandb
        """
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_wandb = use_wandb

        # Teacher is not trainable
        for param in self.teacher.parameters():
            param.requires_grad = False

        # EMA for teacher
        self.teacher_ema = EMA(self.student, decay=teacher_momentum)

        self.global_step = 0
        self.epoch = 0

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

        epoch_losses = {'total': 0.0, 'dino': 0.0, 'koleo': 0.0, 'ibot': 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Get protein structures
            structures = batch['coords']  # List of [n_residues, n_atoms, 3]

            # Forward pass through student
            student_outputs = self.student(structures)  # List of [n_residues, embed_dim]

            # Forward pass through teacher (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher(structures)

            # Compute loss
            # For DINO, we need CLS token outputs
            # Here we'll use global average pooling over residues as a simple approach
            student_cls = [out.mean(dim=0, keepdim=True) for out in student_outputs]
            teacher_cls = [out.mean(dim=0, keepdim=True) for out in teacher_outputs]

            losses = self.criterion(
                student_output=student_cls,
                teacher_output=teacher_cls,
                epoch=self.epoch,
            )

            loss = losses['total']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update teacher with EMA
            self.teacher_ema.update()
            self._copy_params_to_teacher()

            # Logging
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1

            if self.use_wandb and self.global_step % 10 == 0:
                log_dict = {
                    f'train/{key}': value.item()
                    for key, value in losses.items()
                }
                log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                log_dict['train/step'] = self.global_step
                wandb.log(log_dict, step=self.global_step)

            self.global_step += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {self.epoch} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f}")

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def _copy_params_to_teacher(self):
        """Copy EMA parameters from student to teacher."""
        for (name_s, param_s), (name_t, param_t) in zip(
            self.student.named_parameters(), self.teacher.named_parameters()
        ):
            param_t.data.copy_(self.teacher_ema.shadow[name_s])

    def train(self, dataloader, num_epochs: int, save_dir: Optional[str] = None):
        """
        Main training loop.

        Args:
            dataloader: Training dataloader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")

            # Train for one epoch
            epoch_losses = self.train_epoch(dataloader)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Log epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            for key, value in epoch_losses.items():
                print(f"  {key}: {value:.4f}")

            if self.use_wandb:
                wandb.log({
                    f'epoch/{key}': value
                    for key, value in epoch_losses.items()
                }, step=self.global_step)

            # Save checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch + 1}.pt")

        print("\nTraining complete!")

        # Save final model
        if save_dir:
            self.save_checkpoint(save_dir / "final_model.pt")

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'teacher_ema_shadow': self.teacher_ema.shadow,
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.teacher_ema.shadow = checkpoint['teacher_ema_shadow']
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {path}")


def create_model(
    n_atoms: int = 37,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
) -> PackedSequenceEncoder:
    """Create a protein structure encoder model."""
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
    )


def main():
    """Main training function."""
    # Configuration
    config = {
        'n_atoms': 37,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'batch_size': 4,
        'num_epochs': 100,
        'lr': 1e-4,
        'weight_decay': 0.05,
        'teacher_momentum': 0.996,
        'warmup_teacher_temp': 0.04,
        'teacher_temp': 0.07,
        'warmup_teacher_temp_epochs': 30,
        'student_temp': 0.1,
        'pdb_dir': 'CASP14/',
        'file_pattern': 'T*.pdb',
        'save_dir': 'checkpoints',
        'use_wandb': True,
    }

    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project='protein-structure-encoder',
            config=config,
        )

    # Create models
    print("Creating models...")
    student = create_model(
        n_atoms=config['n_atoms'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
    )

    teacher = create_model(
        n_atoms=config['n_atoms'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
    )

    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())

    # Create loss function
    print("Creating loss function...")
    criterion = DINOv2Loss(
        out_dim=config['embed_dim'],
        warmup_teacher_temp=config['warmup_teacher_temp'],
        teacher_temp=config['teacher_temp'],
        warmup_teacher_temp_epochs=config['warmup_teacher_temp_epochs'],
        student_temp=config['student_temp'],
        lambda_koleo=0.0,  # Disable KoLeo for now
        lambda_ibot=0.0,   # Disable iBOT for now
    )

    # Setup teacher temperature schedule
    criterion.dino_loss.setup_teacher_temp_schedule(config['num_epochs'])

    # Create optimizer and scheduler
    print("Creating optimizer...")
    optimizer = AdamW(
        student.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['lr'] * 0.01,
    )

    # Create dataloader
    print("Creating dataloader...")
    transform = RandomRotationTranslation3D(translation_range=0.0)
    dataloader = create_dataloader(
        pdb_dir=config['pdb_dir'],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        transform=transform,
        file_pattern=config['file_pattern'],
    )

    # Create trainer
    print("Creating trainer...")
    trainer = ProteinEncoderTrainer(
        student=student,
        teacher=teacher,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        teacher_momentum=config['teacher_momentum'],
        use_wandb=config['use_wandb'],
    )

    # Train
    print("Starting training...")
    trainer.train(
        dataloader=dataloader,
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir'],
    )

    if config['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main()
