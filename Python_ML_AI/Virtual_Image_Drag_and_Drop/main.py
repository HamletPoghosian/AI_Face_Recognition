import cv2
from cvzone.HandTrackingModule import HandDetector

cam = cv2.VideoCapture(0 ,cv2.CAP_DSHOW)
cam.set(3,1280)
cam.set(4,720)

img1 = cv2.imread('photo/4.jpg')
ox, oy = 400,300
detector = HandDetector(detectionCon=0.5,maxHands=2)

while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType= False)
    h, w, _ = img1.shape
    img[oy: oy+h, ox: ox+w] = img1

    if hands:
        lmList= hands[0]['lmList']
        lenght, info, img = detector.findDistance(lmList[8], lmList[12], img)

        if lenght < 60:
            cursor = lmList[8]
            if ox < cursor[0] < ox + w and oy < cursor[1]< oy + h:
                ox, oy = cursor[0] - w // 2 , cursor[1] - h // 2



    cv2.imshow("Image", img)
    cv2.waitKey(1)

