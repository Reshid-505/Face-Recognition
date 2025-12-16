import cv2
from deepface import DeepFace
import numpy as np
import os

KNOWN_PATH = "known"
THRESHOLD = 0.45

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Üz detektoru
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Tanınmiş üzlərin embedding-ləri
known_embeddings = {}
for file in os.listdir(KNOWN_PATH):
    if file.endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(KNOWN_PATH, file)
        try:
            emb = DeepFace.represent(path, model_name="Facenet512")[0]["embedding"]
            name = os.path.splitext(file)[0]
            known_embeddings[name] = emb
            print("Loaded:", name)
        except Exception as e:
            print("Error loading:", file, e)

video = cv2.VideoCapture(0)

# Kamera full HD
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Kamera açıldı. Q basınca çıxacaq...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Üz aşkarlama
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]

        # DeepFace embedding
        try:
            emb = DeepFace.represent(face_crop, model_name="Facenet512")[0]["embedding"]
        except:
            continue

        best_name = "Unknown"
        best_dist = 99

        # Bütün tanınmış üzlərlə müqayisə
        for name, known_emb in known_embeddings.items():
            dist = cosine_distance(emb, known_emb)
            if dist < best_dist and dist < THRESHOLD:
                best_dist = dist
                best_name = name

        # rəng — tanıyıb tanımamasına görə
        color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)

        # kare və ad yaz
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, best_name, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
