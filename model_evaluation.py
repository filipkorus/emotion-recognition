import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img

def load_dataset(directory):
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        for filename in os.listdir(directory+label):
            image_path = os.path.join(directory, label, filename)
            image_paths.append(image_path)
            labels.append(label)

        # print(label, "Completed")

    return image_paths, labels


def extract_features(images,downsample=False):
    features = []
    for image_path in images:
        img = load_img(image_path, color_mode='grayscale')
        img = np.array(img)

        if downsample:  # affectnet images are 96x96
            img = img[::2, ::2]  # so downsampling to 48x48 is required

        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)

    return features


labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
le = LabelEncoder()
le.fit(labels)

# Ścieżka do folderu z modelami
models_folder = 'models'

# Słownik mapujący nazwy modeli na foldery z danymi testowymi
model_to_data_folder = {
    # '5-layer_affectnet.keras': 'input/affectnet',
    # '5-layer_affectnet_balanced.keras':'input/affectnet_balanced',
	# '5-layer_fer2013.keras':'input/fer2013',
	# '5-layer_fer2013_balanced.keras':'input/fer2013_balanced',
	# '5-layer_merged_dataset.keras':'input/merged_dataset',
	# '5-layer_merged_dataset_balanced.keras':'input/merged_dataset_balanced',
	# '5-layer_mma.keras':'input/mma',
	# '5-layer_mma_balanced.keras':'input/mma_balanced',
	# '5-layer_fer2013_balanced.keras':'input/merged_dataset',
	# '5-layer_affectnet_balanced.keras':'input/merged_dataset',
	# '5-layer_mma_balanced.keras':'input/merged_dataset',

    '5-layer_balanced_filtered_FER2013.keras': 'input/balanced_filtered_FER2013',
    '5-layer_balanced_filtered_FER2013.keras': 'input/merged_dataset',

    'YT_1_balanced_filtered_FER2013.keras': 'input/balanced_filtered_FER2013',
    'YT_1_balanced_filtered_FER2013.keras': 'input/merged_dataset',

    'YT_2_balanced_filtered_FER2013.keras': 'input/balanced_filtered_FER2013',
    'YT_2_balanced_filtered_FER2013.keras': 'input/merged_dataset',

    'YT_3_balanced_filtered_FER2013.keras': 'input/balanced_filtered_FER2013',
    'YT_3_balanced_filtered_FER2013.keras': 'input/merged_dataset',

	#'6-layer_fer2013.keras':'input/fer2013',
	#'6-layer_mma.keras':'input/mma',
}

results = []

# Iteracja po elementach słownika
for model_name, data_folder in model_to_data_folder.items():
    # Wczytanie modelu
    model_path = os.path.join(models_folder, model_name)
    model = load_model(model_path)

    TEST_DIR = f"{data_folder}/test/"
    test = pd.DataFrame()
    test['image'], test['label'] = load_dataset(TEST_DIR)
    test = test.sample(frac=1).reset_index(drop=True)

    test_features = extract_features(test['image'],downsample='affectnet'in data_folder)
    x_test = test_features / 255.0

    y_test = le.transform(test['label'])
    y_test = to_categorical(y_test, num_classes=len(labels))

    # Przewidywanie klas na zestawie testowym
    y_pred_prob = model.predict(x_test,verbose=0)
    y_pred = np.array([np.eye(len(row))[row.argmax()] for row in y_pred_prob])

    # Wygenerowanie raportu klasyfikacji
    report = classification_report(y_test, y_pred)
    print(f"\n\nClassification Report for model {model_name} for {data_folder.replace('input/','')}:")
    print(report)
