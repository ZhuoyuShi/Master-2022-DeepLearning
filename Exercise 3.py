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
plt.imshow(img)
#img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(img_gray)
#%%
# b: Pad the image with a 255 value (white color). Use a pad size of 1
img_pad = cv2.copyMakeBorder(img, 1,1,1,1, cv2.BORDER_CONSTANT, value = [255,255,255])
cv2.imshow("Padded image",img_pad)
cv2.waitKey(0)
#%%
# c: Apply Prewitt operator for edge detection
