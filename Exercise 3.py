# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:38:11 2022

@author: Joyce
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/User/Documents/BDS/B4 Deep learning/Final/triang.png",
                cv2.IMREAD_COLOR)
#plt.imshow(img)

#%%
# b: Pad the image with a 255 value (white color). Use a pad size of 1
img_pad = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_CONSTANT, value = [255,255,255])
img_gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
cv2.imshow("Padded image",img_pad)
cv2.waitKey(0)
#%% package
# c: Apply Prewitt operator for edge detection

#img_gaussian = cv2.GaussianBlur(img_gray,(3,3),0)
#plt.imshow(img_gray)

#convolution
kernelx = np.array([[1,1,1], [0,0,0], [-1,-1,-1]], dtype = int)
kernely = np.array([[1,0,-1], [1,0,-1], [1,0,-1]], dtype = int)
x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)
# turn uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)

plt.imshow(Prewitt, 'gray')
plt.show()
#%% manually
# c: Apply Prewitt operator for edge detection
###################padding manually
pad_img_gray = np.full((img_gray.shape[0]+2, img_gray.shape[1]+2), 255)

for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        pad_img_gray[i+1][j+1] = img_gray[i][j]

#convolution manually
cov_x_img_gray = np.zeros([img_gray.shape[0], img_gray.shape[1]])
cov_y_img_gray = np.zeros([img_gray.shape[0], img_gray.shape[1]])

for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        cov_x_img_gray[i, j] = np.sum(kernelx * pad_img_gray[i:i+3, j:j+3])
        cov_y_img_gray[i, j] = np.sum(kernely * pad_img_gray[i:i+3, j:j+3], dtype = 'uint8')

cov_x_img_gray = np.array(cov_x_img_gray, dtype = 'uint8')     
cov_y_img_gray = np.array(cov_y_img_gray, dtype = 'uint8') 

f = np.array(0.5 * cov_x_img_gray + 0.5 * cov_y_img_gray, dtype = 'uint8') 
plt.imshow(f, "gray")
plt.show()




