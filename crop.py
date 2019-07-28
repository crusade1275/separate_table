# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:32:11 2019

@author: GroverChiu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#---載入欲裁切圖檔
img = cv2.imread("input_35.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #將圖片轉成RGB
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #將圖片轉成灰階

#用Sobel計算x，y方向上的梯度，接著在x方向上减去y方向上的梯度
#藉此，我们留下具有高水平梯度和低垂直梯度的圖。
gradX = cv2.Sobel(img_gray, cv2.CV_16S, dx=1, dy=0, ksize=3)
gradY = cv2.Sobel(img_gray, cv2.CV_16S, dx=0, dy=1, ksize=5)

plt.imshow(gradX, cmap ='gray')
plt.show()
plt.imshow(gradY, cmap ='gray')
plt.show()

gradient = cv2.subtract(gradX, gradY)
plt.imshow(gradient, cmap ='gray')
plt.show()

gradient = cv2.convertScaleAbs(gradient)
plt.imshow(gradient, cmap ='gray')
plt.show()

#----模糊化
blurred = cv2.blur(gradient, (9, 9))
plt.imshow(blurred, cmap ='gray')
plt.show()

(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
plt.imshow(thresh, cmap ='gray')
plt.show()

#填滿噪點
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

plt.imshow(closed, cmap ='gray')
plt.show()

#找出輪廓
(_, contours, hierarchy) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
#在輪廓畫上標記
cv2.drawContours(img, [box], -1, (0, 255, 0), 3)

plt.imshow(img)
plt.show()

#將標記處裁切
min(box[:,1])
rmin=min(box[:,1])
cmin=min(box[:,0])
rmax=max(box[:,1])
cmax=max(box[:,0])
#print(rmin,cmin,rmax,cmax)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img2= img
img2 = np.zeros((int(-(rmin-rmax)),int(-(cmin-cmax)),3))
img2.shape
for row in range(rmin,rmax):
    for col in range(cmin,cmax):
        for chal in range(3):
            img2[row-rmin,col-cmin,chal] = img[row,col,chal]

#---輸出裁切好的圖檔
cv2.imwrite( 'output_.png', img2, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

plt.imshow(img2)
plt.show()
