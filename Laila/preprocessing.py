import numpy as np
import cv2
def process(img):
    img[img<150]=0
    img[img>0]=255
    img[55-2:55+2,75-2:75+2]=100
    img=img/255
    img=img[55-20:55+20,75-20:75+20]
    return img