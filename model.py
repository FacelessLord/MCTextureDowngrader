import torch_rocm_win as torch_rocm
torch = torch_rocm.torch
nn = torch.nn


# print(f"PyTorch Version: {torch.__version__}")
# print(f"ROCm Version: {torch.version.hip}")
# device = torch.device("hip" if torch.version.hip else "cpu")
# print(f"Using device: {device}")
#
# if torch.version.hip:
#     tensor = torch.randn(1000, 1000, device=device)
#     print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device=device)/1024/1024:.2f} MB")
class DownsampleCNN_v1(nn.Module):
    def __init__(self):
        super(DownsampleCNN_v1, self).__init__()

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

        # Upsampling layers (8x8 -> 16x16)
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Output layer
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalize output to [-1, 1] range
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upconv1(x)
        x = self.out(x)
        return x

class DownsampleCNN_v2(nn.Module):
    def __init__(self):
        super(DownsampleCNN_v2, self).__init__()

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

        # Upsampling layers (8x8 -> 16x16)
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Upsampling layers (8x8 -> 16x16)
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=9, stride=1, padding=4),
            nn.Tanh()  # Normalize output to [-1, 1] range
        )

        # Output layer
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalize output to [-1, 1] range
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.out(x)
        return x

class UpsampleCNN(nn.Module):
    def __init__(self):
        super(UpsampleCNN, self).__init__()

        # Processing layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 16x16 -> 8x8
        )

        # Upsampling layers (8x8 -> 16x16)
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Upsampling layers (8x8 -> 16x16)
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=9, stride=1, padding=4),
            nn.Tanh()  # Normalize output to [-1, 1] range
        )

        # Output layer
        self.out = nn.Sequential(
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalize output to [-1, 1] range
        )

    def forward(self, x):
        x = self.conv2(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.out(x)
        return x