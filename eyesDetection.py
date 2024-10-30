import cv2
import numpy as np
import dlib


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


# Göz noktalarını çizme fonksiyonu
def draw_eye_landmarks(frame, landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

    for (x, y) in left_eye:
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    for (x, y) in right_eye:
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)


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

        # Göz noktalarını çiz
        draw_eye_landmarks(frame, landmarks)

        # Sol ve sağ göz için göz oranlarını hesapla
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Göz durumunu tahmin et
        ear_threshold = 0.15
        left_label = "Kapalı" if left_ear < ear_threshold else "Açık"
        right_label = "Kapalı" if right_ear < ear_threshold else "Açık"

        # Sonuçları ekranda göster
        cv2.putText(frame, f"Sol Göz: {left_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Sağ Göz: {right_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Açık veya Kapalı", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
