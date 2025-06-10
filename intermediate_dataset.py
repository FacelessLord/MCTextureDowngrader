import json
import os

from PIL import Image
from torch.utils.data import Dataset


class IntermediateImageDataset(Dataset):
    def __init__(self, in_folder, out_folder, transform=None):
        print(f"Loading {in_folder}:{out_folder}")
        if not os.path.exists(in_folder) or not os.path.exists(out_folder):
            raise FileNotFoundError(in_folder +" or "+ out_folder)


        self.transform = transform

        # Get list of input images
        self.inputs = [os.path.join(in_folder, file) for file in os.listdir(in_folder)]

        # Verify expected images exist
        self.expected = [os.path.join(out_folder, file) for file in os.listdir(in_folder)]

        # Ensure we have matching pairs
        assert len(self.inputs) == len(self.expected), \
            f"Mismatched datasets: {len(self.inputs)} inputs vs {len(self.expected)} expected"

        # Sort to maintain pairing
        self.inputs.sort()
        self.expected.sort()
        print(f"Loaded  {in_folder}:{out_folder}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Load input image
        input_img_path = self.inputs[idx]
        input_image = Image.open(input_img_path).convert('RGBA')

        # Load expected output image
        expected_img_path = self.expected[idx]
        expected_image = Image.open(expected_img_path).convert('RGBA')

        # Apply transforms
        if self.transform:
            input_image_transformed = self.transform(input_image)
            expected_image_transformed = self.transform(expected_image)
        else:
            input_image_transformed = input_image
            expected_image_transformed = expected_image

        return {
            'input': input_image_transformed,
            'expected': expected_image_transformed
        }
