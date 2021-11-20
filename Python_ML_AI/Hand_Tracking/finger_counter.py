import cv2
import time
import os

wCam , hCam = 1080, 1080
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderName = "photo"
imageList = os.listdir(folderName)
overlayList = []
for imPath in imageList:
    image = cv2.imread(f'{folderName}/{imPath}')
    overlayList.append((image))

while True:
    success , img = cap.read()

    h, w, c = overlayList[1].shape
    img[0:h-500, 0:w-500] = overlayList[1]


    cv2.imshow("Image", img)
    cv2.waitKey(1)