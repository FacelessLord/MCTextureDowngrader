import os

import torch
from PIL import Image
from torchvision import transforms

from model import DownsampleCNN_v2, UpsampleCNN

def load_images():
    images = []
    for file in os.listdir('manual_input'):
        if not file.endswith('.png'):
            continue

        with Image.open(os.path.join('manual_input', file)) as image:
            transform = transforms.ToTensor()
            images.append((file, transform(image.convert('RGBA'))))

    return images

def manual_check():
    model = DownsampleCNN_v2()
    model.load_state_dict(
        torch.load('best_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    device = torch.device("hip" if torch.version.hip else "cpu")
    model = model.to(device)
    model.eval()
    upsample_model = UpsampleCNN()
    upsample_model.load_state_dict(
        torch.load('best_upsample_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    device = torch.device("hip" if torch.version.hip else "cpu")
    upsample_model = upsample_model.to(device)
    upsample_model.eval()

    images = load_images()

    if not os.path.exists('manual_output'):
        os.mkdir('manual_output')

    for (file, image) in images:
        image = image.to(device)

        with torch.no_grad():
            # Get prediction
            output = model(image)
            output = upsample_model(output)

            # Convert back to PIL image
            output_image = transforms.ToPILImage()(output.squeeze(0))

        output_image.save(os.path.join('manual_output', file), format='PNG', bits=8)

manual_check()