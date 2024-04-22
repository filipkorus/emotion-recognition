import os
import shutil
from PIL import Image


def resize_image(input_path, output_path, size=(48, 48)):
    with Image.open(input_path) as img:
        img_resized = img.resize(size, Image.LANCZOS)
        img_resized.save(output_path)


def merge_datasets(dataset_dirs, output_dir):
    # Ensure output directory exists
    if not os.path.exists('input/' + output_dir):
        os.makedirs('input/' + output_dir)

    # Loop through each dataset directory
    for dataset_dir in dataset_dirs:
        # Loop through valid, test, and train directories
        for split in ['valid', 'test', 'train']:
            # Loop through label1, label2, and label3 directories
            for label in ['neutral', 'sad', 'happy', 'surprise', 'angry', 'fear', 'disgust']:
                # Define source and destination directories
                src_dir = os.path.join('input/' + dataset_dir, split, label)
                dest_dir = os.path.join('input/' + output_dir, split, label)

                # Copy and resize files from source to destination directory
                if os.path.exists(src_dir):
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)
                    for filename in os.listdir(src_dir):
                        src_file = os.path.join(src_dir, filename)
                        dest_file = os.path.join(dest_dir, filename)

                        # Resize images in affectnet to 48x48
                        if 'affectnet' in dataset_dir:
                            resize_image(src_file, dest_file)
                        else:
                            shutil.copy(src_file, dest_file)
                    print(f"Copied files from {src_dir} to {dest_dir}")


if __name__ == '__main__':
    # Define dataset directories
    dataset_dirs = ['fer2013', 'affectnet', 'mma']

    # Define output directory
    output_dir = 'merged_dataset'

    # Merge datasets
    merge_datasets(dataset_dirs, output_dir)
