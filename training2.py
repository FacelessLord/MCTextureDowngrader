import os.path

from torch import GradScaler

import torch_rocm_win as torch_rocm
from intermediate_dataset import IntermediateImageDataset

torch = torch_rocm.torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model import UpsampleCNN

saved_model_name = 'best_upsample_model.pth'

class TeacherStudentTrainer:
    def __init__(self, student_model, teacher_model, device='cuda'):
        self.student = student_model.to(device)
        self.teacher = teacher_model.to(device)
        self.device = device

        # Freeze teacher weights
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Set up loss functions
        self.mse_loss = nn.MSELoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

        # Optimizer for student
        self.optimizer = optim.Adam(self.student.parameters(), lr=1e-4)
        self.scaler = GradScaler()

    def train_step(self, batch):
        inputs = batch['input'].to(self.device)
        expected = batch['expected'].to(self.device)

        # Forward passes
        student_output = self.student(inputs)

        # Calculate losses
        reconstruction_loss = self.mse_loss(student_output, expected)
        distillation_loss = self.mse_loss(student_output, self.teacher(inputs))

        # Total loss with weighting
        total_loss = reconstruction_loss + 0.5 * distillation_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.requires_grad_(True)
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'distillation_loss': distillation_loss.item()
        }


def train_teacher_student(teacher_model, student_model, dataset, check_dataset, batch_size=32, epochs=25):
    print("Prepare train_teacher_student")
    device = torch.device("hip" if torch.version.hip else "cpu")
    trainer = TeacherStudentTrainer(student_model, teacher_model, device)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(check_dataset, batch_size=batch_size, shuffle=False)

    print("Training loop")
    best_loss = validate(trainer, val_loader)['total_loss']
    print(f'Best loss on start: {best_loss}')

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_losses = train_epoch(trainer, dataloader)

        with torch.no_grad():
            val_losses = validate(trainer, val_loader)

        # Print results
        print("\nEpoch Summary:")
        print(f"Training Loss: {train_losses['total_loss']:.4f}")
        print(f"Validation Loss: {val_losses['total_loss']:.4f}")

        # Save best model based on validation loss
        if best_loss is not None and val_losses['total_loss'] < best_loss:
            torch.save(student.state_dict(), saved_model_name)
        best_loss = val_losses['total_loss']


def train_epoch(trainer, loader):
    running_losses = {'total_loss': 0., 'reconstruction_loss': 0.,
                      'distillation_loss': 0.}
    total_batches = len(loader)

    for i, batch in enumerate(loader):
        losses = trainer.train_step(batch)
        for key in running_losses:
            running_losses[key] += losses[key]

        if (i + 1) % 100 == 0:
            current_batch = i + 1
            print(f"Batch {current_batch}/{total_batches}, "
                  f"Loss: {running_losses['total_loss'] / current_batch:.4f}")

    return {k: v / total_batches for k, v in running_losses.items()}


def validate(trainer, loader):
    running_losses = {'total_loss': 0., 'reconstruction_loss': 0.,
                      'distillation_loss': 0.}

    with torch.no_grad():
        for batch in loader:
            losses = trainer.train_step(batch)
            for key in running_losses:
                running_losses[key] += losses[key]

    return {k: v / len(loader) for k, v in running_losses.items()}


# Example usage
teacher = UpsampleCNN()  # Your model architecture
student = UpsampleCNN()

if os.path.exists(saved_model_name):
    student.load_state_dict(
        torch.load(saved_model_name, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    teacher.load_state_dict(
        torch.load(saved_model_name, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

device = torch.device("hip" if torch.version.hip else "cpu")

# Initialize teacher weights randomly
torch.manual_seed(42)
teacher.apply(lambda m: torch.nn.init.xavier_normal_(
    m.weight) if isinstance(m, nn.Conv2d) else None)

dataset = IntermediateImageDataset('train_intermediate', 'output', transforms.ToTensor())
check_dataset = IntermediateImageDataset('check_intermediate', 'output', transforms.ToTensor())
# Train
train_teacher_student(teacher, student, dataset, check_dataset)
