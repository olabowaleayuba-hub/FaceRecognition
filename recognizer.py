import cv2 as cv
import os

# -------------------------
# Load Face Detector
# -------------------------

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# Load Trained Model
# -------------------------

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# -------------------------
# Load Labels
# -------------------------

labels = {}

with open("labels.txt", "r") as file:
    for line in file:
        label, name = line.strip().split(",")
        labels[int(label)] = name

# -------------------------
# Open Webcam
# -------------------------

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

# -------------------------
# Recognition Loop
# -------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = labels[label]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        cv.putText(
            frame,
            f"{name}",
            (x, y-10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (x, y+h+25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
