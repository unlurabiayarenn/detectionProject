import os
import cv2
import numpy as np
import dlib
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Göz oranı hesaplamak için kullanılan fonksiyon
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Üst alt mesafe
    B = np.linalg.norm(eye[2] - eye[4])  # İç dış mesafe
    C = np.linalg.norm(eye[0] - eye[3])  # Yatay mesafe
    ear = (A + B) / (2.0 * C)
    return ear

# Dlib ile yüz ve göz algılayıcıyı yükleme
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Veri yükleme fonksiyonu
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    labels = df['state'].values  # 'state' sütununu kullanıyoruz
    images = []  # Görsel veriler burada oluşturulacak

    # Göz durumlarına göre sahte görüntüler oluşturma
    for label in labels:
        if label == 'open':
            img = np.ones((26, 34, 1))  # Açık göz için beyaz görüntü
        else:
            img = np.zeros((26, 34, 1))  # Kapalı göz için siyah görüntü

        images.append(img)

    return np.array(images), labels

# CNN Modelini Oluşturma
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(34, 26, 1)))  # Giriş şekli
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Veri setini yükle
csv_file = '/Users/rabiayarenunlu/PycharmProjects/detectionProject/dataset_eye_on_off.csv'  # CSV dosyasının yolu
X, y = load_data(csv_file)

# Etiketleri sayısal değerlere dönüştür
y = np.where(y == 'open', 1, 0).astype(np.float32)  # "open" = 1, "close" = 0

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = create_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Kameradan görüntü alma ve veri kaydetme
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı! Kamerayı kontrol edin.")
    exit()

# Kayıt dizinleri
save_dir = '/Users/rabiayarenunlu/PycharmProjects/detectionProject/data/train'  # Kayıt dizininin yolu
os.makedirs(os.path.join(save_dir, 'closed'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'open'), exist_ok=True)

count_closed = 0
count_open = 0

# Ana döngü
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı! ret değeri:", ret)
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Sol göz (36-42 numaralı noktalar)
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        # Sağ göz (42-48 numaralı noktalar)
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Göz oranlarını hesapla
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Göz oranlarını kullanarak tahmini değerlendir
        ear_threshold = 0.25  # Eşik değeri

        left_label = "Kapalı" if left_ear < ear_threshold else "Açık"
        right_label = "Kapalı" if right_ear < ear_threshold else "Açık"

        # Göz görüntülerini kes ve işle
        eye_offset = 10  # Göz alanını genişletmek için bir offset belirle

        # Sol göz için
        left_eye_img = frame[
                       max(left_eye[1][1] - eye_offset, 0): left_eye[5][1] + eye_offset,
                       max(left_eye[0][0] - eye_offset, 0): left_eye[3][0] + eye_offset
                       ]

        # Sağ göz için
        right_eye_img = frame[
                        max(right_eye[1][1] - eye_offset, 0): right_eye[5][1] + eye_offset,
                        max(right_eye[0][0] - eye_offset, 0): right_eye[3][0] + eye_offset
                        ]

        # Görüntüleri yeniden boyutlandır ve normalleştir
        left_eye_img = cv2.resize(left_eye_img, (34, 26))
        left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)
        left_eye_img = left_eye_img.reshape((26, 34, 1)) / 255.0

        right_eye_img = cv2.resize(right_eye_img, (34, 26))
        right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)
        right_eye_img = right_eye_img.reshape((26, 34, 1)) / 255.0

        # Model ile tahmin et
        left_prediction = model.predict(np.expand_dims(left_eye_img, axis=0))
        right_prediction = model.predict(np.expand_dims(right_eye_img, axis=0))

        # Tahminleri değerlendir
        left_label_model = "Kapalı" if left_prediction < 0.5 else "Açık"
        right_label_model = "Kapalı" if right_prediction < 0.5 else "Açık"

        # Görüntüyü kaydet
        if left_label == "Kapalı":
            cv2.imwrite(os.path.join(save_dir, 'closed', f'closed_{count_closed}.jpg'), frame)
            count_closed += 1
        if left_label == "Açık":
            cv2.imwrite(os.path.join(save_dir, 'open', f'open_{count_open}.jpg'), frame)
            count_open += 1

        # Sonuçları ekranda göster
        cv2.putText(frame, f"Sol Göz: {left_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Sağ Göz: {right_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Göz çevresindeki noktaları çiz:
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Açık veya Kapalı", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
