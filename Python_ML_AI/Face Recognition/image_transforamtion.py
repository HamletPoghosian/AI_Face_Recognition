import cv2 as cv
import numpy as np
import tensorflow

img = cv.imread('Photos\Cat\_111434467_gettyimages-1143489763.jpg')
cv.imshow('Cat', img)

def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

transform = translate(img,100,100)
cv.imshow('Transform', transform)

# Rotation

def rotate(img, angel, rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMap = cv.getRotationMatrix2D(rotPoint,angel,1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMap, dimensions)

rotateImg = rotate(img,45)
cv.imshow('rotate', rotateImg)

# flipping

flip = cv.flip(img,1)
cv.imshow('Flip', flip)



cv.waitKey(0)