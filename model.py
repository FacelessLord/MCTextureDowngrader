import torch.nn as nn


# print(f"PyTorch Version: {torch.__version__}")
# print(f"ROCm Version: {torch.version.hip}")
# device = torch.device("hip" if torch.version.hip else "cpu")
# print(f"Using device: {device}")
#
# if torch.version.hip:
#     tensor = torch.randn(1000, 1000, device=device)
#     print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device=device)/1024/1024:.2f} MB")
class DownsampleCNN(nn.Module):
    def __init__(self):
        super(DownsampleCNN, self).__init__()

        # Input layer (32x32 RGBA -> 64 channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 32x32 -> 16x16
        )

        # Processing layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 16x16 -> 8x8
        )

        # Upsampling layers
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Output layer (8x8 -> 16x16)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Normalize output to [-1, 1] range
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upconv1(x)
        x = self.out(x)
        return x

# Initialize the model
model = DownsampleCNN()

# Print model summary
print(model)