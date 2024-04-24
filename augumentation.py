import os
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
import random

# Funkcja do przetwarzania obrazów w podfolderze i zastosowania augmentacji danych
def augment_images_in_folder(folder_path, target_num_images):
    # Lista nazw plików w folderze
    image_files = os.listdir(folder_path)
    num_images = len(image_files)

    # Sprawdzenie, czy liczba obrazów przekracza docelową liczbę obrazów
    if num_images < target_num_images:
        # Obliczenie liczby obrazów do wygenerowania dla tej klasy emocji
        num_augmented_images = target_num_images - num_images

        # Tworzenie sekwencji augmentacji danych
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # Odbicie lustrzane z prawdopodobieństwem 0.5
            iaa.Affine(rotate=(-20, 20)),  # Losowe obracanie o kąt od -20 do 20 stopni
            iaa.GaussianBlur(sigma=(0, 2.0))  # Dodanie rozmycia Gaussa
        ])

        # Przetwarzanie każdego obrazu w folderze
        for i in range(num_augmented_images):
            # Losowe wybranie obrazu z folderu
            filename = random.choice(image_files)
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # Zastosowanie augmentacji danych
            image_aug = seq(image=np.array(image))
            augmented_image = Image.fromarray(image_aug)  # Konwersja z numpy.ndarray na PIL.Image

            # Zapisanie augmentowanego obrazu w tym samym folderze
            augmented_image_path = os.path.join(folder_path, f'augmented_{i}_{filename}')
            augmented_image.save(augmented_image_path)

        print(f"Data augumentation in {folder_path} finished")


if __name__ == '__main__':
    dataset_path = 'input/merged_dataset_2'

    labels = os.listdir(os.path.join(dataset_path, 'train'))

    folders_and_target_numbers_of_imgs = [
        ('train', 42_000),
        ('test', 6_000),
        ('valid', 12_000),
    ]

    for folder, target_num_images in folders_and_target_numbers_of_imgs:
        for label in labels:
            emotion_folder_path = os.path.join(dataset_path, folder, label)
            augment_images_in_folder(emotion_folder_path, target_num_images)
