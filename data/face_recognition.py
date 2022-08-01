import cv2
import os
from PIL import Image
import shutil

# This script is working, but false positives and negatives

"""
Copy all square images from validation to sqr_pretrain
"""
face_rec = True  # Do copy function?

if face_rec:
    folder = 'D:/Lucha_Data/datasets/NSD/images/valid/'
    count = 0
    cascPath = "C:/Users/david/Python/deepReconPyTorch/data/haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)

    folder_out = 'D:/Lucha_Data/datasets/NSD/images/faces/'

    for imagePath in os.listdir(folder):
        imagePath = os.path.join(folder, imagePath)
        # Read the image
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        draw = False

        if draw:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Faces found", image)
            cv2.waitKey(0)

        if len(faces) > 0:
            print("In image, {}, cv2 found {} faces!".format(imagePath, len(faces)))
            shutil.copy(imagePath, os.path.join(folder_out, 'faces/'))
            count += 1
        else:
            shutil.copy(imagePath, os.path.join(folder_out, 'non_faces/'))


    print('Final count of images with faces {}'.format(count))

