import os
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
import random
import shutil

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
            # Obrót (Rotation): Obracanie obrazu o losowy kąt, np. w zakresie od -20 do 20 stopni.
            iaa.Affine(rotate=(-20, 20)),
            
            # Skalowanie (Scaling): Zmiana rozmiaru obrazu, powiększanie lub zmniejszanie go o pewien procent.
            iaa.Affine(scale={"x": (0.95, 1.2), "y": (0.95, 1.2)}),  # Skalowanie w zakresie 80%-120%
            
            # Odbicie (Flip): Odbicie obrazu w poziomie z prawdopodobieństwem 0.5.
            iaa.Fliplr(0.5),
            
            # Zmiana jasności (Brightness Adjustment): Zwiększanie lub zmniejszanie jasności obrazu.
            iaa.Multiply((0.8, 1.2)),  # Jasność zmieniana w zakresie 80%-120%
            
            # Dodawanie szumu (Adding Noise): Dodawanie DELIKATNEGO szumu Gaussa.
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Szum Gaussa o skali 0-12.75 (przy założeniu 8-bitowych obrazów)
            
            # Blurring (Rozmywanie): Stosowanie filtrów rozmywających, takich jak DELIKATNE rozmycie Gaussa.
            iaa.GaussianBlur(sigma=(0, 1.0)),  # Rozmycie Gaussa z sigma w zakresie 0-1.0
            
            # Zmiana kontrastu (Contrast Adjustment): Zwiększanie lub zmniejszanie kontrastu obrazu.
            iaa.LinearContrast((0.8, 1.2)),  # Kontrast zmieniany w zakresie 80%-120%
            
            # Zmiana nasycenia (Saturation Adjustment): Zwiększanie lub zmniejszanie nasycenia kolorów obrazu.
            # iaa.MultiplySaturation((0.8, 1.2))  # Nasycenie zmieniane w zakresie 80%-120%
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
    original_dataset_path = 'input/merged_dataset'
    balanced_dataset_path = 'input/merged_dataset_balanced'

    shutil.copytree(original_dataset_path, balanced_dataset_path)

    labels = os.listdir(os.path.join(balanced_dataset_path, 'train'))

    folders_and_target_numbers_of_imgs = [
        ('train', 49_000),
        ('test', 7_000),
        ('valid', 14_000),
    ]

    for folder, target_num_images in folders_and_target_numbers_of_imgs:
        for label in labels:
            emotion_folder_path = os.path.join(balanced_dataset_path, folder, label)
            augment_images_in_folder(emotion_folder_path, target_num_images)
