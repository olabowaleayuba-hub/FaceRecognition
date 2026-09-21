import cv2
import os

person_name = input("Enter the person's name: ").strip()

save_path = os.path.join("faces", person_name)

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(cv2.data.haarcascades)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv.imshow("collect_faces", gray)
   
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(
            frame,
            (x,y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        face = gray[y:y+h, x:x+w]
        face= cv2.resize(face, (200,200 ))

        filename = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(filename, face)

        count += 1
        if count>= 100:
            break

        cv2.putText(
            frame,
            f"images: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    cv2.imshow("Collect Faces", frame)
    if count >= 100:
        break
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    
    
    
cap.release()
cv2.destroyAllWindows()

print(f"\ncollected {count} images for {person_name}")

        




