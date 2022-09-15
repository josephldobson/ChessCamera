import cv2 as cv

cam = cap = cv.VideoCapture('http://192.168.1.132:8080/video')

cv.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    h = frame.shape[0]
    w = frame.shape[1]
    fra = frame.copy()
    cv.line(fra, (int(w/2),0), (int(w/2),h), (0,0,255),1)
    cv.line(fra, (0,int(h*0.8)), (w, int(h*0.8)), (0,255,0), 1)
    if not ret:
        print("failed to grab frame")
        break
    cv.imshow("test", fra)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv.destroyAllWindows()