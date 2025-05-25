import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.v2.functional import pil_to_tensor

from dataset import ImageDataset
from model import DownsampleCNN


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

    def train_step(self, batch):
        inputs, _ = batch  # Assuming second item is label we don't need
        inputs = inputs.to(self.device)

        # Forward passes
        student_output = self.student(inputs)
        teacher_output = self.teacher(inputs)

        # Calculate losses
        reconstruction_loss = self.mse_loss(student_output, inputs)
        distillation_loss = self.mse_loss(student_output, teacher_output)

        # Total loss with weighting
        total_loss = reconstruction_loss + 0.5 * distillation_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'distillation_loss': distillation_loss.item()
        }


def train_teacher_student(teacher_model, student_model, dataset, batch_size=32, epochs=10):
    print("Prepare train_teacher_student")
    device = torch.device("hip" if torch.version.hip else "cpu")
    trainer = TeacherStudentTrainer(student_model, teacher_model, device)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Training loop")
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train step
        running_losses = {'total_loss': 0., 'reconstruction_loss': 0.,
                          'distillation_loss': 0.}
        total_batches = len(dataloader)

        for i, batch in enumerate(dataloader):
            losses = trainer.train_step(batch)

            # Accumulate losses
            for key in running_losses:
                running_losses[key] += losses[key]

            # Print progress
            if (i + 1) % 100 == 0:
                current_batch = i + 1
                print(f"Batch {current_batch}/{total_batches}, "
                      f"Loss: {running_losses['total_loss'] / current_batch:.4f}")

        # Average losses for epoch
        avg_losses = {k: v / total_batches for k, v in running_losses.items()}
        print("\nEpoch averages:")
        for loss_name, value in avg_losses.items():
            print(f"{loss_name}: {value:.4f}")


# Example usage
teacher = DownsampleCNN()  # Your model architecture
student = DownsampleCNN()

# Initialize teacher weights randomly
torch.manual_seed(42)
teacher.apply(lambda m: torch.nn.init.xavier_normal_(
    m.weight) if isinstance(m, nn.Conv2d) else None)

dataset = ImageDataset('train_index.txt', pil_to_tensor)
check_dataset = ImageDataset('check_index.txt', pil_to_tensor)
# Train
train_teacher_student(teacher, student, dataset)