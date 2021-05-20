import numpy as np
import face_recognition
import cv2
import os

path = 'KnownFaces'
images = []
className = []
myList = os.listdir(path)

print(myList)
