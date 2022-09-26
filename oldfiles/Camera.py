import cv2 as cv
import numpy as np
import time
import scipy as sp
from linefinder import *
from movetracker import *

cap = cv.VideoCapture('http://192.168.0.34:8080/video')
while(True):
    ret, frame = cap.read()
    h = frame.shape[0]
    frame = frame[int(h * 0.3):h, :]
    frame = cv.resize(frame, (1600,800), interpolation=cv.INTER_AREA)
    #lines_h, lines_v, frame = find_lines(frame)
    cv.waitKey(1)
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()