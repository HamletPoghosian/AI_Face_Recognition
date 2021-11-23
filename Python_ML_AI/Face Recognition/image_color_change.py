import cv2 as cv

# show image
img = cv.imread('Photos\Cat\_111434467_gettyimages-1143489763.jpg')
cv.imshow('Cat', img)

# convert to cray
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(img,125,175)
cv.imshow('Canny', canny)

cv.waitKey(0)