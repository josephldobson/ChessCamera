import cv2 as cv
import numpy as np
import math
from FindLines2 import find_lines_P
from FindLines import find_lines

cap = cv.VideoCapture('http://192.168.1.132:8080/video')
while(True):
    ret, frame = cap.read()
    frame = find_lines(frame)
    cv.waitKey(1)
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()