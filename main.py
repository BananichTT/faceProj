import numpy as np
import face_recognition
import cv2
import os

path = 'KnownFaces'
images = []
className = []
List = os.listdir(path)

print(List)

for cls in List:
    curlImg = cv2.imread(f'{path}/{cls}')
    images.append(curlImg)
    className.append(os.path.splitext(cls)[0])

print(className)
