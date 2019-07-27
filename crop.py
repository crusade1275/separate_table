# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:32:11 2019

@author: GroverChiu
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#---輸入欲裁切圖檔
img = cv2.imread("input_10.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gradX = cv2.Sobel(img_gray, cv2.CV_16S, dx=1, dy=0, ksize=11)
gradY = cv2.Sobel(img_gray, cv2.CV_16S, dx=0, dy=1, ksize=13)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

#----模糊化
blurred = cv2.blur(gradient, (9, 9))
blurred = cv2.GaussianBlur(gradient, (5,5),0)
blurred = cv2.medianBlur(gradient,5)

(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)


(_, contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
cv2.drawContours(img_gray, [box], -1, (0, 255, 0), 3)


cv2.imshow('Image_img_gray', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

min(box[:,1])
rmin=min(box[:,1])
cmin=min(box[:,0])
rmax=max(box[:,1])
cmax=max(box[:,0])

#print(rmin,cmin,rmax,cmax)
img2= img

plt.imshow(img_gray)
img2 = np.zeros((int(-(rmin-rmax)),int(-(cmin-cmax)),3))
img2.shape
for row in range(rmin,rmax):
    for col in range(cmin,cmax):
        for chal in range(3):
            img2[row-rmin,col-cmin,chal] = img[row,col,chal]

#---輸出已裁切圖檔
cv2.imwrite( 'crop_output_.png', img2, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

plt.imshow(img2)
