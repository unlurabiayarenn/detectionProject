import os
import cv2
import numpy as np
import dlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd

# Göz oranı hesaplama fonksiyonu
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Üst alt mesafe
    B = np.linalg.norm(eye[2] - eye[4])  # İç dış mesafe
    C = np.linalg.norm(eye[0] - eye[3])  # Yatay mesafe
    ear = (A + B) / (2.0 * C)
    return ear

# Dlib ile yüz ve göz algılayıcıyı yükleme
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Modeli oluşturma fonksiyonu
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(26, 34, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 'Kapalı' ve 'Açık' için 2 sınıf
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Veri yükleme fonksiyonu
def load_data(data_dir):
    images = []
    labels = []
    for label in ['open', 'closed']:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (34, 26))  # Resmi 34x26 boyutuna getir
            images.append(img)
            labels.append(0 if label == 'closed' else 1)  # 'closed' -> 0, 'open' -> 1
    return np.array(images), np.array(labels)

# Eğitim verilerini yükleyin
X, y = load_data('data/train')
X = X.reshape(-1, 26, 34, 1)  # Boyutlandırma

# Modeli oluştur
model = create_model()

# Modeli eğitme
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Ağırlıkları kaydet (sadece son katmanın ağırlıkları)
last_layer_weights = model.layers[-1].get_weights()
flat_weight = last_layer_weights[0].flatten()  # Sadece ağırlıkları al
df = pd.DataFrame(flat_weight)
df.to_csv('last_layer_weights.csv', index=False)

# Kameradan görüntü alma
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı! Kamerayı kontrol edin.")
    exit()

# Ana döngü
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Sol ve sağ göz için göz görüntülerini kes
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Göz görüntülerini daha geniş bir alandan kes
        left_eye_image = gray[max(0, landmarks.part(36).y - 10):landmarks.part(41).y + 10,
                               max(0, landmarks.part(36).x - 10):landmarks.part(39).x + 10]
        right_eye_image = gray[max(0, landmarks.part(42).y - 10):landmarks.part(47).y + 10,
                                max(0, landmarks.part(42).x - 10):landmarks.part(45).x + 10]

        # Göz görüntülerini 26x34 boyutuna getir
        left_eye_image = cv2.resize(left_eye_image, (34, 26))
        right_eye_image = cv2.resize(right_eye_image, (34, 26))

        left_eye_image = np.expand_dims(left_eye_image, axis=-1)
        right_eye_image = np.expand_dims(right_eye_image, axis=-1)

        # Modelin tahmini
        left_pred = model.predict(np.expand_dims(left_eye_image, axis=0))
        right_pred = model.predict(np.expand_dims(right_eye_image, axis=0))

        # Tahminlerin güven eşiğini kontrol et
        confidence_threshold = 0.7  # Eşiği artırdık
        left_confidence = np.max(left_pred)
        right_confidence = np.max(right_pred)

        left_label = "Kapali" if np.argmax(left_pred) == 0 and left_confidence > confidence_threshold else "Acik"
        right_label = "Kapali" if np.argmax(right_pred) == 0 and right_confidence > confidence_threshold else "Acik"

        # Göz noktalarını çizme
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Sonuçları ekranda göster
        cv2.putText(frame, f"Sol Göz: {left_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Sag Göz: {right_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Acik veya Kapali", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
