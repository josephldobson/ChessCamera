import cv2 as cv
import math
import numpy as np
import cv2 as cv

def transform(img,pts,w,h):
    rows,cols,ch = img.shape
    pts1 = np.float32([[1275,1080],[2885,1090],[387,1964],[3860,2024]])
    pts2 = np.float32([[100,100],[500,100],[100,900],[500,900]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img,M,(1000,500))
    return(dst)
