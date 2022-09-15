import cv2 as cv
import numpy as np
from Transformation import transform
from FindLines import find_lines
from FindLines2 import find_lines_P
img = cv.imread('OldImages/Test2/signal-2022-04-02-152546_004.jpeg')
after = find_lines(img)
cv.imshow('after', after)
cv.waitKey(0)
