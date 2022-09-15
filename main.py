import cv2 as cv
import numpy as np
import time
import scipy as sp

def resize_and_process(img, size):
    h = img.shape[0]
    original = img[int(h*0.3):h, :]
    img = cv.resize(original, size, interpolation=cv.INTER_AREA)
    out = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    out = cv.blur(out, (3, 3))
    out = cv.Canny(out, 70, 210)
    return original, img, out


def hough_line_vertical(img):
    lines = cv.HoughLines(img, rho=1, theta=np.pi/360, threshold=100, srn=0, stn=0, min_theta=-1, max_theta=1)
    lines = lines[:40]
    lines = np.reshape(lines, (-1, 2))
    return lines


def hough_line_horizontal(img):
    lines = cv.HoughLines(img, rho=1, theta=np.pi/360, threshold=100, min_theta=np.pi/2-0.3, max_theta=np.pi/2+0.3)
    lines = lines[:50]
    lines = np.reshape(lines, (-1, 2))
    return lines


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def new_horizontal_lines_after_rotation(lines, angle, centre_point):
    b = -np.cos(angle)
    a = -np.sin(angle)
    c = -a * centre_point[0] + b * centre_point[1]
    r = abs(c) / (a**2 + b**2)**0.5
    x0 = intersection_polar([r, np.pi/2-angle], [0, np.pi/2])[0]
    points = [intersection_polar(i, [r, np.pi/2 - angle]) for i in lines]
    lines = np.array([[(((x0-i[0])**2+(0-i[1])**2)**0.5), np.pi/2] for i in points])
    return lines


def intersection_polar(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    a = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(a, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def intersection_of_lines(lines):
    points = np.array([intersection_polar(i, j) for i in lines for j in lines if i[1] != j[1]])
    return points


def gaussian_mode_of_points(points):
    # TODO improve disappearing point selection

    canvas = np.zeros((800, 800))
    for i in points:
        x = i[0]-400
        y = i[1]+400
        if 0 < x < 800 and 0 < y < 800:
            canvas[x, y] += 1
    blurred = sp.ndimage.gaussian_filter(canvas, sigma=1)
    point = np.unravel_index(blurred.argmax(), blurred.shape)
    x, y = point
    return [x+400, y-400]


def distance_point_and_line(point, line):
    rho, theta = line
    m, n = point
    a = np.cos(theta)
    b = np.sin(theta)
    c = -a * a * rho - b * b * rho
    d = abs(a * m + b * n + c) / ((a ** 2 + b ** 2) ** 0.5)
    return d


def create_new_line(point, angle):
    b = -np.cos(angle)
    a = -np.sin(angle)
    c = a * point[1] + b * point[0]
    r = abs(c) / (a**2 + b**2)**0.5
    return [r, angle]


def filter_vertical_lines(point, lines):
    # find disappearing point and filter lines passing through
    distances = np.array([distance_point_and_line(point, i) for i in lines])
    order = np.argsort(distances)
    lines = np.array(lines)[order]
    distances = distances[order]
    lines = lines[:14]
    distances = distances[:14]
    lines = lines[distances < 6]
    order = np.argsort(lines[:, 1])
    lines = lines[order]

    # filter wrong vertical lines
    tan_lines = [np.tan(x) for x in lines[:, 1]]
    diffs = [tan_lines[i+1]-tan_lines[i] for i in range(len(tan_lines)-1)]
    median = np.median([i for i in diffs if 0.1 < i < 0.5])
    member = [sum([1 if abs(median-abs(i-j)) < 0.02 else 0 for i in tan_lines]) for j in tan_lines]
    lines = np.array([lines[i] for i in range(len(lines)) if member[i] != 0])
    tan_lines = np.array([tan_lines[i] for i in range(len(tan_lines)) if member[i] != 0])

    # filter duplicate vertical lines and add missing lines
    # TODO improve line selection
    new_lines = [lines[0]]
    for i in range(len(lines)-1):
        if abs(abs(tan_lines[i+1]-tan_lines[i])-median) < 0.04:
            new_lines.append(lines[i+1])
            continue
    lines = np.array(new_lines)

    # find most likely mid-point and indices of left and right lines
    smallest_angle_index = np.argmin(np.array([abs(i[1]) for i in lines]))
    smallest_angle = lines[smallest_angle_index][1]
    if not 0.1 < smallest_angle < 0.1:
        smallest_angle = 0

    # TODO create linear regression based prediction and line selection
    left_line = lines[len(lines) - 1]
    left_index = round(np.tan(smallest_angle - left_line[1]) / median) + 4
    right_line = lines[0]
    right_index = round(np.tan(smallest_angle - right_line[1]) / median) + 4
    point = intersection_polar(left_line, right_line)
    tan_gradient = (np.tan(left_line[1]) - np.tan(right_line[1])) / (right_index-left_index)
    leftmost_tan = np.tan(left_line[1])+tan_gradient*left_index

    lines = [create_new_line(point, np.arctan(i)) for i in np.linspace(leftmost_tan, (leftmost_tan-8*tan_gradient), 9)]
    return lines


def filter_horizontal_lines(lines):
    mode = sp.stats.mode(lines[:, 1], keepdims=True)[0]
    lines = lines[abs(lines[:, 1]-mode) <= 0.01]
    return mode, lines


def refilter_horizontal_lines(lines_h, lines_v, dims):
    # calculate crossing lines of vertical and horizontal lines
    intersections = np.array([intersection_polar(i, j) for i in lines_v for j in lines_h])
    canvas = np.zeros((800, 1600), np.uint8)
    for i in intersections:
        x = i[1]
        y = i[0]
        if (0 < x < dims[1]) and (0 < y < dims[0]):
            canvas[x, y] += 50
    canvas = sp.ndimage.gaussian_filter(canvas, sigma=1)
    cross_lines = cv.HoughLines(canvas, rho=1, theta=np.pi/360, threshold=10,
                                min_theta=np.pi+np.pi/3, max_theta=np.pi+np.pi/2-0.1)
    cross_lines = np.reshape(cross_lines, (-1, 2))
    cross_line = cross_lines[0]

    new_horizontal = np.array([[intersection_polar(cross_line, j)[1], np.pi/2] for j in lines_v])

    return new_horizontal


def crossing_points(lines_v, lines_h):
    crossing = np.array([[intersection_polar(i, j) for i in lines_v] for j in lines_h])
    return crossing


def draw_lines(lines, image, thick, iterator, col=(100, 100, 0)):
    for i in iterator:
        rho = lines[i][0]
        theta = lines[i][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
        cv.line(image, pt1, pt2, col, thick, cv.LINE_AA)
    return


def transform(lines_v, lines_h, original):

    orig_h = original.shape[0]
    orig_w = original.shape[1]
    left_line = lines_v[0]
    right_line = lines_v[len(lines_v)-1]
    top_line = lines_h[len(lines_h)-1]
    bottom_line = lines_h[0]
    top, bottom, left, right = 100, 900, 100, 900
    if bottom_line[0] > 800:
        bottom_line = lines_h[1]
        top = 200

    if bottom_line[0] < 600:
        top_line = lines_h[len(lines_h)-2]
        bottom = 800

    tl = intersection_polar(left_line, top_line)
    tr = intersection_polar(right_line, top_line)
    bl = intersection_polar(left_line, bottom_line)
    br = intersection_polar(right_line, bottom_line)
    tl = [tl[0]*orig_w/1600, tl[1]*orig_h/800]
    tr = [tr[0]*orig_w/1600, tr[1]*orig_h/800]
    bl = [bl[0]*orig_w/1600, bl[1]*orig_h/800]
    br = [br[0]*orig_w/1600, br[1]*orig_h/800]
    pts1 = np.float32([tl, tr, bl, br])
    pts2 = np.float32([[left, top], [right, top], [left, bottom], [right, bottom]])
    print(pts1)
    print(pts2)
    m = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(original, m, (1000, 1000))
    for i in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        cv.line(dst, [0, i], [1000, i], (255, 0, 0), 1, cv.LINE_AA)
        cv.line(dst, [i, 0], [i, 1000], (0, 255, 0), 1, cv.LINE_AA)
    return dst



def display_in_grid(img):

    original = cv.imread(img)
    original, resized, test_photo = resize_and_process(original, (1600, 800))

    lines_h = hough_line_horizontal(test_photo)
    mode, lines_h = filter_horizontal_lines(lines_h)
    lines_h = new_horizontal_lines_after_rotation(lines_h, mode[0], [800, 400])
    test_photo = rotate_image(test_photo, mode[0]*180/np.pi-90)
    resized = rotate_image(resized, mode[0]*180/np.pi-90)

    lines_v = hough_line_vertical(test_photo)
    points = intersection_of_lines(lines_v)
    point = gaussian_mode_of_points(points)
    lines_v = filter_vertical_lines(point, lines_v)

    lines_h = refilter_horizontal_lines(lines_h, lines_v, (1600, 800))
    original = transform(lines_v, lines_h, original)
    cv.imshow('ok',original)
    cv.waitKey(0)
    return resized
    # cv.imshow('resized', resized)
    # cv.waitKey(0)


def photos():
    st = time.time()
    display_in_grid('chessboard_photos/opencv_frame_0.png')
    display_in_grid('chessboard_photos/chessboard4.jpg')
    display_in_grid('chessboard_photos/chessboard3.jpg')
    display_in_grid('chessboard_photos/chess_game/position1.jpg')
    display_in_grid('chessboard_photos/chess_game/position2.jpg')
    display_in_grid('chessboard_photos/chess_game/position3.jpg')
    display_in_grid('chessboard_photos/chess_game/position4.jpg')
    display_in_grid('chessboard_photos/chess_game/position5.jpg')
    display_in_grid('chessboard_photos/chess_game/position6.jpg')
    display_in_grid('chessboard_photos/chess_game/position7.jpg')
    display_in_grid('chessboard_photos/chess_game/position8.jpg')
    display_in_grid('chessboard_photos/chess_game/position9.jpg')
    display_in_grid('chessboard_photos/chess_game/position10.jpg')
    display_in_grid('chessboard_photos/chess_game/position11.jpg')
    display_in_grid('chessboard_photos/chess_game/position12.jpg')
    display_in_grid('chessboard_photos/chess_game/position13.jpg')
    display_in_grid('chessboard_photos/chess_game/position14.jpg')
    display_in_grid('chessboard_photos/chess_game/position15.jpg')
    display_in_grid('chessboard_photos/chess_game/position16.jpg')
    display_in_grid('chessboard_photos/chess_game/position17.jpg')
    display_in_grid('chessboard_photos/chess_game/position18.jpg')
    display_in_grid('chessboard_photos/chess_game/position19.jpg')
    display_in_grid('chessboard_photos/chess_game/position20.jpg')
    display_in_grid('chessboard_photos/chess_game/position21.jpg')
    display_in_grid('chessboard_photos/chess_game/position22.jpg')
    display_in_grid('chessboard_photos/chess_game/position23.jpg')
    display_in_grid('chessboard_photos/chess_game/position24.jpg')
    display_in_grid('chessboard_photos/chess_game/position25.jpg')
    print('Execution time:', (time.time() - st)/27, 'seconds')
    return


photos()
