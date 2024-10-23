import cv2
import dlib

# Göz oranı hesaplamak için kullanılan fonksiyon
#def eye_aspect_ratio(eye):
#    A = ((eye[1][0] - eye[5][0]) ** 2 + (eye[1][1] - eye[5][1]) ** 2) ** 0.5
#    B = ((eye[2][0] - eye[4][0]) ** 2 + (eye[2][1] - eye[4][1]) ** 2) ** 0.5
#    C = ((eye[0][0] - eye[3][0]) ** 2 + (eye[0][1] - eye[3][1]) ** 2) ** 0.5
#    ear = (A + B) / (2.0 * C)
#  return ear


def eye_aspect_ratio(eye):
    #göz çevresindeki belirli noktaları kullan
    X = ((eye[1][0] - eye[5][0]) ** 2 + (eye[1][1] - eye[5][1]) ** 2) ** 0.5  # Üst alt mesafe
    Y = ((eye[2][0] - eye[4][0]) ** 2 + (eye[2][1] - eye[4][1]) ** 2) ** 0.5  # İç dış mesafe
    Z = ((eye[0][0] - eye[3][0]) ** 2 + (eye[0][1] - eye[3][1]) ** 2) ** 0.5  # Yatay mesafe
    ear = (X + Y) / (2.0 * Z)

    # Göz oranını ayarlamak için ekleyebileceğin opsiyonel bir faktör
    adjustment_factor = 1.0  # Göz yapına göre ayarlayabilirsin
    return ear * adjustment_factor


# Dlib ile yüz ve göz algılayıcıyı yükleme
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Göz oranı eşik değeri
EYE_AR_THRESH = 0.30  # Göz açık/kapalı ayrımı için eşik

# Kameradan görüntü alma
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Sol göz (36-41 numaralı noktalar)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        # Sağ göz (42-47 numaralı noktalar)
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Sol ve sağ göz için göz oranını hesapla
        left_eye_ratio = eye_aspect_ratio(left_eye)
        right_eye_ratio = eye_aspect_ratio(right_eye)

        # Gözlerin açık mı kapalı mı olduğunu kontrol et
        if left_eye_ratio < EYE_AR_THRESH:
            cv2.putText(frame, "Sol Goz Kapali", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Sol Goz Acik", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if right_eye_ratio < EYE_AR_THRESH:
            cv2.putText(frame, "Sag Goz Kapali", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Sag Goz Acik", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Göz çevresindeki noktaları çiz
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow("Open or Close", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
