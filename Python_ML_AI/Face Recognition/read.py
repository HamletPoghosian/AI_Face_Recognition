import  cv2 as cv

# reading image
img = cv.imread('Photos\Cat\_111434467_gettyimages-1143489763.jpg')

cv.imshow('Cat', img)
cv.waitKey(0)

# reading video

capture = cv.VideoCapture('Photos/Video/open_cv_test.mp4')

while True:
    isTrue, frame = capture.read()

    cv.imshow('video', frame)

    if cv.waitKey(20) & 0xFF == ord('m'):
        break

capture.release()
cv.destroyAllWindows()

