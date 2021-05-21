import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime

path = 'KnownFaces' # путь до папки с фотографиями
images = []
classNames = []
List = os.listdir(path)
print(List)

# добавление информации
for cls in List:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

# функция отвечающая за декодирование фотографий
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # перевод фотографии в удобный формат для распознования
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# функция каоторая записывает в файл время появления и имя человека в кадре
def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name}, {dtString}')

encodeListKnown = findEncodings(images) # переменная отвечающая за обработанные фотографии
print("Декодирование закончено")

cap = cv2.VideoCapture(0) # подключение веб-камеры

# цикл для работы веб-камеры
while True:
    success, img = cap.read() # отвечают за 1 кадр из видео
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # переменная принимающая подготовленный кадр для обработки из видео
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # поиск всех лиц в текущем кадре
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # цикл для распознавания
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) # отвечает за соответствие к уже известному программе лицу
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) # вероятность совпадения
        matchIndex = np.argmin(faceDis)

        # проверка на известные лица
        if matches[matchIndex]:
            name = classNames[matchIndex]

            # рамка вокруг лица человека с его именем
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow("WebCam", img) # окно выводящее видео
    cv2.waitKey(1)