import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.src.legacy.preprocessing.image import ImageDataGenerator

OUTPUT_FILE = 'classification_report.txt'

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
le = LabelEncoder()
le.fit(labels)

# Ścieżka do folderu z modelami
models_folder = 'models'

# Słownik mapujący nazwy modeli na foldery z danymi testowymi
model_to_data_folder = {
    '5-layer_affectnet.keras':'input/affectnet',
    '5-layer_merged_dataset.keras':'input/merged_dataset',
    '5-layer_merged_dataset_balanced.keras':'input/merged_dataset_balanced',
    '5-layer_balanced_filtered_FER2013.keras': 'input/balanced_filtered_FER2013',
    '5-layer_balanced_filtered_FER2013_aug.keras': 'input/balanced_filtered_FER2013_aug',

	'YT_5_affectnet.keras':'input/affectnet',
    'YT_5_merged_dataset.keras':'input/merged_dataset',
    'YT_5_merged_dataset_balanced.keras':'input/merged_dataset_balanced',
    'YT_5_balanced_filtered_FER2013.keras': 'input/balanced_filtered_FER2013',
    'YT_5_balanced_filtered_FER2013_aug.keras': 'input/balanced_filtered_FER2013_aug',
}

results = []

# Iteracja po elementach słownika
for model_name, data_folder in model_to_data_folder.items():
    # Wczytanie modelu
    model_path = os.path.join(models_folder, model_name)
    model = load_model(model_path)

    TEST_DIR = f"{data_folder}/test/"
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        color_mode='grayscale',
        target_size=(48, 48),
        batch_size=128,
        class_mode='categorical',
        shuffle=True
    )

    x_test = []
    y_test = []

    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]
        x_test.extend(x_batch)
        y_test.extend(np.argmax(y_batch, axis=1))

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Wygenerowanie raportu klasyfikacji
    report = classification_report(y_test, y_pred, target_names=labels)

    print(f'{model_name} done')

    print(f"\n\nClassification Report for model {model_name} for {data_folder.replace('input/','')}:", file=open(OUTPUT_FILE, 'a'))
    print(report, file=open(OUTPUT_FILE, 'a'))
