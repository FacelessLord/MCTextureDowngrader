from PIL import Image
import os
import json
import shutil

from id_holder import IdHolder
from imagePopulator import populate_images


# - Walk over all files in `faithful` directory
# - filter by 32 x32 size
# - move images into `input` folder, under name `${dirname}__{filename}`
# - for each file in input find corresponding file in `mc` folder and move this file into output folder

def find_png_files_by_dimensions(directory, width=None, height=None) -> list[tuple[str, str, (int, int)]]:
    """
    Find PNG files in directory matching specified dimensions.

    Args:
        directory: Path to search for PNG files
        width: Target width in pixels (optional)
        height: Target height in pixels (optional)

    Returns:
        List of tuples containing (filename, joined_name, dimensions)
    """
    matching_files = []

    try:
        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith('.png'):
                    filepath = os.path.join(dirpath, filename)
                    new_name = dirpath.replace(directory + "\\", '').replace('\\', "__") + "__" + filename

                    if os.path.isfile(filepath):
                        try:
                            with Image.open(filepath) as img:
                                w, h = img.size

                                # Check dimension conditions
                                if width is not None and height is not None:
                                    if w == width and h == height:
                                        matching_files.append((filepath, new_name, (w, h)))
                                elif width is not None:
                                    if w == width:
                                        matching_files.append((filepath, new_name, (w, h)))
                                elif height is not None:
                                    if h == height:
                                        matching_files.append((filepath, new_name, (w, h)))

                        except IOError:
                            print(f"Could not read image: {filepath}")

    except FileNotFoundError:
        print(f"Directory '{directory}' not found")
    except PermissionError:
        print(f"No permission to access directory '{directory}'")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

    return matching_files


def find_minecraft_textures(faithful_textures: list[tuple[str, str, (int, int)]]) -> dict[str, str]:
    mc_textures = find_png_files_by_dimensions('mc', 16, 16)
    index = {mc_name: mc_path for (mc_path, mc_name, _) in mc_textures}

    faith_to_mc_map = dict()
    for (faithful_path, faithful_name, _) in faithful_textures:
        if faithful_name in index:
            faith_to_mc_map[faithful_path] = index[faithful_name]
        else:
            print('[Warn] Not found mc texture: ' + faithful_name)

    return faith_to_mc_map


def load_index():
    if os.path.exists('index.txt'):
        with open('index.txt', 'rt') as index_file:
            return json.load(index_file)

    faithful_textures = find_png_files_by_dimensions('faithful', 32, 32)
    src_index = find_minecraft_textures(faithful_textures)

    with open('index.txt', 'wt') as index_file:
        json.dump(src_index, index_file, indent=2)

    return src_index


def prepare_data():
    index = load_index()

    # population multiplication coefficient
    k = populate_images()

    faith_check_dir, faith_dir, mc_check_dir, mc_dir = prepare_data_directories()

    move_and_populate_data(index, k, faith_check_dir, faith_dir, mc_check_dir, mc_dir)


def move_and_populate_data(index, k, faith_check_dir, faith_dir, mc_check_dir, mc_dir):
    id_holder = IdHolder()
    check_data_start = int(len(index) * 9 * k / 10)
    is_check_data = False
    id = 0

    train_index = dict()
    check_index = dict()
    for faith_path in index:
        mc_path = index[faith_path]
        all_images = populate_image(faith_path, mc_path)

        faith_save_dir = faith_check_dir if is_check_data else faith_dir
        mc_save_dir = mc_check_dir if is_check_data else mc_dir
        save_index = check_index if is_check_data else train_index
        for (faith_img, mc_img) in all_images:
            id = id_holder.next()
            faith_dataset_path = os.path.join(faith_save_dir, f"_{id}.png")
            mc_dataset_path = os.path.join(mc_save_dir, f"_{id}.png")
            faith_img.save(faith_dataset_path, bits=8)
            mc_img.save(mc_dataset_path, bits=8)

            save_index[faith_dataset_path] = mc_dataset_path
        if id > check_data_start:
            is_check_data = True

    with open('train_index.txt', 'wt') as index_file:
        json.dump(train_index, index_file, indent=2)
    with open('check_index.txt', 'wt') as index_file:
        json.dump(check_index, index_file, indent=2)


def populate_image(faith_path, mc_path):
    images = []
    with Image.open(faith_path, 'r') as in_img:
        with Image.open(mc_path, 'r') as out_img:
            populated_images = populate_images(in_img, out_img)
            images.extend(populated_images)
    return images


def prepare_data_directories():
    faith_dir = os.path.join('.', 'input')
    if not os.path.exists(faith_dir):
        os.mkdir(faith_dir)
    mc_dir = os.path.join('.', 'output')
    if not os.path.exists(mc_dir):
        os.mkdir(mc_dir)
    faith_check_dir = os.path.join('.', 'check_input')
    if not os.path.exists(faith_check_dir):
        os.mkdir(faith_check_dir)
    mc_check_dir = os.path.join('.', 'check_output')
    if not os.path.exists(mc_check_dir):
        os.mkdir(mc_check_dir)
    return faith_check_dir, faith_dir, mc_check_dir, mc_dir


prepare_data()
