import json
import os

import torch
from PIL import Image
from torchvision import transforms

from model import DownsampleCNN_v2


def load_images(index_path):
    with open(index_path, 'rt') as index_file:
        index = json.load(index_file)

    for file in index:
        with Image.open(file) as image:
            transform = transforms.ToTensor()
            yield (file, transform(image.convert('RGBA')))



def intermediate_prepare():
    model = DownsampleCNN_v2()
    model.load_state_dict(
        torch.load('best_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    device = torch.device("hip" if torch.version.hip else "cpu")
    model = model.to(device)
    model.eval()

    if not os.path.exists('train_intermediate'):
        os.mkdir('train_intermediate')
    if not os.path.exists('check_intermediate'):
        os.mkdir('check_intermediate')

    for (file, image) in load_images('train_index.txt'):
        image = image.to(device)

        with torch.no_grad():
            # Get prediction
            output = model(image)
            # Convert back to PIL image
            output_image = transforms.ToPILImage()(output.squeeze(0)).convert('RGBA', colors=8).convert('P')

        save_path = 'train_intermediate'
        output_image.save(os.path.join(save_path, file.replace('.\\input\\', '')), format='PNG', bits=8)


intermediate_prepare()
