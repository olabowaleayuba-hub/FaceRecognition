import cv2 as cv
print(cv.__version__)
print(cv.data.haarcascades)

#test
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print(face_cascade.empty())

