import  cv2 as cv

# function for resize video and image . Work for  image , video , live Vide
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# function of change size only Live Video
def changeRes(width,height):
    capture.set(3,width)
    capture.set(4,height)

# reading image
img = cv.imread('Photos\Cat\_111434467_gettyimages-1143489763.jpg')
rescale_img = rescaleFrame(img)
cv.imshow('Cat', img)
cv.imshow('Cat resize', rescale_img)
cv.waitKey(0)

# reading video

capture = cv.VideoCapture('Photos/Video/open_cv_test.mp4')



while True:
    isTrue, frame = capture.read()

    cv.imshow('video', frame)

    re_size_frame = rescaleFrame(frame,scale=0.70)
    cv.imshow('video_resize', re_size_frame)

    if cv.waitKey(20) & 0xFF == ord('m'):
        break

capture.release()
cv.destroyAllWindows()

# test git

print("test git")
print("test git")