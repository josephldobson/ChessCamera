import cv2 as cv
import numpy as np
import time
import scipy as sp
from linefinder import *
from movetracker import *

cam = cv.VideoCapture(0)

cv.namedWindow("test")

img_counter = 0
ret, oldframe = cam.read()
board = chess.Board()
while True:


    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    oldframe = np.copy(frame)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed

        board = test_image(oldframe, frame, board)
        oldframe = np.copy(frame)
        img_counter += 1
        cv.imshow("test", frame)
        cv.waitKey(0)

cam.release()

cv.destroyAllWindows()