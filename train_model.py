import cv2 as cv
import os
import numpy as np

faces = []
labels = []
names = {}

label_id = 0
dataset = "faces"

for person in os.listdir(dataset):
    person_path = os.path.join(dataset, person)

    if not os.path.isdir(person_path):
        continue
    names[label_id] = person

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(label_id)

    label_id += 1

    recognizer = cv.face.LBPHFaceRecognizer_create()

    recognizer.train(faces, np.array(labels))

    recognizer.save("trainer.yml")

    with open("labels.txt", "w") as f:
        for id, name in names.items():
            f.write(f"{id}, {name}\n")

    print("Training complete!")

#print(cv.data.haarcascades)