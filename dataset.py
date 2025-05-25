import json
import os

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, index_path, transform=None):
        print(f"Loading {index_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)
        with open(index_path, 'rt') as index_file:
            index = json.load(index_file)

        self.transform = transform

        # Get list of input images
        self.inputs = [key for key in index]

        # Verify expected images exist
        self.expected = [index[key] for key in index]

        # Ensure we have matching pairs
        assert len(self.inputs) == len(self.expected), \
            f"Mismatched datasets: {len(self.inputs)} inputs vs {len(self.expected)} expected"

        # Sort to maintain pairing
        self.inputs.sort()
        self.expected.sort()
        print(f"Loaded {index_path}")

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
