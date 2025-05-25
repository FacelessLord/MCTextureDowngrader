import numpy as np
from PIL import Image, ImageOps, ImageEnhance


def mirror_image(in_image, out_image) -> tuple[Image, Image]:
    return (ImageOps.mirror(in_image), ImageOps.mirror(out_image))

def brighten_one(image: Image, factor):
    rgb = image.convert('RGB')
    alpha = image.split()[-1].convert('L')

    enhancer = ImageEnhance.Brightness(rgb)
    brightened_rgb = enhancer.enhance(factor)

    return Image.merge('RGBA', (*brightened_rgb.split(), alpha))

def create_lighten_images(factor):
    def lighten_images(in_image, out_image) -> tuple[Image, Image]:
        return (brighten_one(in_image, factor), brighten_one(out_image, factor))

    return lighten_images

def apply_populators(input_list: list[tuple[Image, Image]] | int, *populators):
    if isinstance(input_list, int):
        return input_list * (1+len(populators))

    update = []
    for populator in populators:
        for (in_image, out_image) in input_list:
            (new_in_image, new_out_image) = populator(in_image, out_image)
            update.append((new_in_image, new_out_image))

    input_list.extend(update)
    return input_list

def populate_images(in_image=None, out_image=None) -> list[tuple[Image, Image]] | int:
    results = [(in_image, out_image)] if in_image is not None else 1
    results = apply_populators(results, mirror_image)
    results = apply_populators(results, create_lighten_images(1.1), create_lighten_images(1.5))
    return results