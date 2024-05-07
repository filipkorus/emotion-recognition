import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

model = load_model('models/5-layer_merged_dataset.keras')

print_propabilities = False

# Label encoder
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
le = LabelEncoder()
le.fit(labels)

# OpenCV text settings
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 255, 0)  # color in BGR
thickness = 1
lineType = cv2.LINE_AA

# Window settings
window_title = 'Emotion recognition'

# Variables for tracking ROI
last_roi = None
frames_without_face = 0
max_frames_without_face = 10  # Number of frames to wait before resetting ROI


def emotion_recognition(face_img):
    face_img = face_img / 255.0

    pred = model.predict(face_img.reshape(1, 48, 48, 1), verbose=0)
    pred_label = le.inverse_transform([pred.argmax()])[0]

    if print_propabilities:
        print(f'propabilities: {pred[0]} => {pred_label}')

    return pred_label, pred.max()


def detect_face(frame, padding=20):
    global last_roi, frames_without_face

    face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face = None

    if len(faces_frontal) == 0:
        faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces_frontal) == 0:
            frames_without_face += 1
            if frames_without_face >= max_frames_without_face:
                last_roi = None

            return None, None
        else:
            face = faces_profile[0]
    else:
        face = faces_frontal[0]

    frames_without_face = 0

    x, y, w, h = face

    # Adjust ROI based on the average position and size of detected faces
    x_with_padding = max(0, x - padding)
    y_with_padding = max(0, y - padding - 20)  # Przesunięcie o dodatkowe 20 pikseli w górę

    w_with_padding = w + 2 * padding
    h_with_padding = h + 2 * padding

    roi = (x_with_padding, y_with_padding, w_with_padding, h_with_padding)

    if last_roi is not None:
        # Smooth transition of ROI
        last_x, last_y, last_w, last_h = last_roi
        x_with_padding = int((x_with_padding + last_x) / 2)
        y_with_padding = int((y_with_padding + last_y) / 2)
        w_with_padding = int((w_with_padding + last_w) / 2)
        h_with_padding = int((h_with_padding + last_h) / 2)

    last_roi = roi

    face_image = gray[y_with_padding:y_with_padding + h_with_padding, x_with_padding:x_with_padding + w_with_padding]
    resized_face_image = cv2.resize(face_image, (48, 48))

    return last_roi, resized_face_image


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi, resized_face_image = detect_face(frame, padding=30)

        if roi is not None:
            (x, y, w, h) = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label, prop = emotion_recognition(resized_face_image)
            cv2.putText(frame, f'{label} ({round(prop * 100, 2)}%)', (x + 2, y - 10), font, fontScale, color, thickness,
                        lineType)

        cv2.imshow(window_title, frame)

        if cv2.waitKey(1) == -1 and cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
