import os
import shutil
import random
from unzip import unzip

datasets = ['affectnet', 'fer2013', 'mma']

for dataset in datasets:
    unzip(dataset)

    # Define the directories
    data_dir = f'data/{dataset}/'

    train_dir = f'input/{dataset}/train'
    test_dir = f'input/{dataset}/test'
    valid_dir = f'input/{dataset}/valid'

    # Create train, test, and valid directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # List all label classes
    label_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Define the train, test, and valid ratios
    train_ratio = 0.7
    test_ratio = 0.10
    valid_ratio = 0.20

    # Store the total number of images
    total_images = 0

    # Iterate over each label class
    for label_class in label_classes:
        # Create directories for each label class in train, test, and valid sets
        os.makedirs(os.path.join(train_dir, label_class), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label_class), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, label_class), exist_ok=True)

        # Get all images for the current label class
        images = os.listdir(os.path.join(data_dir, label_class))

        # Update the total number of images
        total_images += len(images)

        # Shuffle the images
        random.shuffle(images)

        # Calculate split indices
        num_images = len(images)
        train_split = int(train_ratio * num_images)
        test_split = int((train_ratio + test_ratio) * num_images)

        # Split the images
        train_images = images[:train_split]
        test_images = images[train_split:test_split]
        valid_images = images[test_split:]

        # Copy images to train, test, and valid directories
        for image in train_images:
            shutil.copy(os.path.join(data_dir, label_class, image), os.path.join(train_dir, label_class, image))

        for image in test_images:
            shutil.copy(os.path.join(data_dir, label_class, image), os.path.join(test_dir, label_class, image))

        for image in valid_images:
            shutil.copy(os.path.join(data_dir, label_class, image), os.path.join(valid_dir, label_class, image))

        # Display the number of images in each folder
        print(f"Number of images in {label_class}:")
        print(f"Train: {len(train_images)} ({(len(train_images)/num_images)*100:.2f}%)")
        print(f"Test: {len(test_images)} ({(len(test_images)/num_images)*100:.2f}%)")
        print(f"Valid: {len(valid_images)} ({(len(valid_images)/num_images)*100:.2f}%)\n")

    # Display the total number of images
    print(f"Total number of images: {total_images}")

    print(f"{dataset} splitting completed!")
