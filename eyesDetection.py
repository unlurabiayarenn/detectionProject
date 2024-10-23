import os
import cv2
import numpy as np
import dlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input


# Göz oranı hesaplamak için kullanılan fonksiyon
def eye_aspect_ratio(eye):
    A = ((eye[1][0] - eye[5][0]) ** 2 + (eye[1][1] - eye[5][1]) ** 2) ** 0.5  # Üst alt mesafe
    B = ((eye[2][0] - eye[4][0]) ** 2 + (eye[2][1] - eye[4][1]) ** 2) ** 0.5  # İç dış mesafe
    C = ((eye[0][0] - eye[3][0]) ** 2 + (eye[0][1] - eye[3][1]) ** 2) ** 0.5  # Yatay mesafe
    ear = (A + B) / (2.0 * C)
    return ear

# Dlib ile yüz ve göz algılayıcıyı yükleme
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Veri yükleme fonksiyonu
def load_data(data_dir):
    images = []
    labels = []
    for label in ['closed', 'open']:
        label_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Boyutlandır
            images.append(img)
            labels.append(0 if label == 'closed' else 1)  # Kapalı için 0, açık için 1
    return np.array(images), np.array(labels)

# CNN Modelini Oluşturma
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # İlk katman olarak Conv2D ekleniyor
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
train_dir = '/Users/rabiayarenunlu/PycharmProjects/detectionProject/data/train'  # Eğitim verinizin dizini
X, y = load_data(train_dir)
X = X / 255.0  # Normalize et
y = y.astype(np.float32)

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = create_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Kameradan görüntü alma ve veri kaydetme
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamerayı açamadım! Lütfen kamerayı kontrol edin.")
    exit()

# Kayıt dizinleri
save_dir = '/Users/rabiayarenunlu/PycharmProjects/detectionProject/data/train'
os.makedirs(os.path.join(save_dir, 'closed'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'open'), exist_ok=True)

count_closed = 0
count_open = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı! ret değeri:", ret)
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Sol göz (36-41 numaralı noktalar)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        # Sağ göz (42-47 numaralı noktalar)
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Göz görüntülerini kes
        #left_eye_img = frame[left_eye[1][1]:left_eye[5][1], left_eye[0][0]:left_eye[3][0]]
        #right_eye_img = frame[right_eye[1][1]:right_eye[5][1], right_eye[0][0]:right_eye[3][0]]

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
        left_eye_img = cv2.resize(left_eye_img, (64, 64)) / 255.0
        right_eye_img = cv2.resize(right_eye_img, (64, 64)) / 255.0

        # Model ile tahmin et
        left_prediction = model.predict(np.expand_dims(left_eye_img, axis=0))
        right_prediction = model.predict(np.expand_dims(right_eye_img, axis=0))

        # Tahminleri değerlendir
        left_label = "Kapali" if left_prediction < 0.5 else "Acik"
        right_label = "Kapali" if right_prediction < 0.5 else "Acik"

        # Görüntüyü kaydet
        # Görüntüyü kaydet
        if left_label == "Kapali":
            cv2.imwrite(os.path.join(save_dir, 'closed', f'closed_{count_closed}.jpg'), frame)

        if left_label == "Acik":
            cv2.imwrite(os.path.join(save_dir, 'open', f'open_{count_open}.jpg'), frame)

        cv2.putText(frame, f"Sag Goz: {left_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Sol Goz: {right_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Göz çevresindeki noktaları çiz
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Open or Close", frame)

    # Kayıt için durumu kontrol et
    #if count_closed >= 50 and count_open >= 50:
        #print("Yeterli görüntü kaydedildi!")
        #break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
