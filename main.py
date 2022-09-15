import cv2 as cv
import numpy as np
import time
import scipy as sp


def resize_and_process(img, size):
    """Returns the image trimmed to 1/3 0f the original height, then resized to 'size', then with the canny algorithm"""

    h = img.shape[0]
    original = img[int(h*0.3):h, :]
    img = cv.resize(original, size, interpolation=cv.INTER_AREA)
    out = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    out = cv.blur(out, (3, 3))
    out = cv.Canny(out, 70, 210)
    return original, img, out


def hough_line_vertical(img):
    """Returns vertical lines in input image using the houghlines algorithm."""

    lines = cv.HoughLines(img, rho=1, theta=np.pi/360, threshold=100, srn=0, stn=0, min_theta=-1, max_theta=1)
    lines = lines[:40]
    lines = np.reshape(lines, (-1, 2))
    return lines


def hough_line_horizontal(img):
    """Returns horizontal lines in input image using the houghlines algorithm."""

    lines = cv.HoughLines(img, rho=1, theta=np.pi/360, threshold=100, min_theta=np.pi/2-0.3, max_theta=np.pi/2+0.3)
    lines = lines[:50]
    lines = np.reshape(lines, (-1, 2))
    return lines


def rotate_image(image, angle):
    """Rotates input image by input angle."""

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def new_horizontal_lines_after_rotation(lines, angle, centre_point):
    """Returns new output lines after being rotated around a centre-point."""

    b = -np.cos(angle)
    a = -np.sin(angle)
    c = -a * centre_point[0] + b * centre_point[1]
    r = abs(c) / (a**2 + b**2)**0.5
    x0 = intersection_polar([r, np.pi/2-angle], [0, np.pi/2])[0]
    points = [intersection_polar(i, [r, np.pi/2 - angle]) for i in lines]
    lines = np.array([[(((x0-i[0])**2+(0-i[1])**2)**0.5), np.pi/2] for i in points])
    return lines


def intersection_polar(line1, line2):
    """Returns the intersection of two lines given in polar form."""

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
    """Returns the intersection points of all lines given as a set."""

    points = np.array([intersection_polar(i, j) for i in lines for j in lines if i[1] != j[1]])
    return points


def gaussian_mode_of_points(points):
    """Given an array of 2D points, applies a Gaussian blur, and outputs the largest point."""

    # TODO finetune point selection
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
    """Returns the distance between a point and a line, with the line given in polar."""

    rho, theta = line
    m, n = point
    a = np.cos(theta)
    b = np.sin(theta)
    c = -a * a * rho - b * b * rho
    d = abs(a * m + b * n + c) / ((a ** 2 + b ** 2) ** 0.5)
    return d


def create_new_line(point, angle):
    """Creates a new line in polar form, given a cartesian input point and angle in radians"""

    b = -np.cos(angle)
    a = -np.sin(angle)
    c = a * point[1] + b * point[0]
    r = abs(c) / (a**2 + b**2)**0.5
    return [r, angle]


def filter_vertical_lines(point, lines):
    """1) Filters the vertical lines by distance to their dissappearing point.
    2) Removes duplicate lines
    3) Uses correct lines to create a set of fine-tunes lines.
    4) Returns all 9 vertical lines from the chess board.
    """

    # 1)
    distances = np.array([distance_point_and_line(point, i) for i in lines])
    order = np.argsort(distances)
    lines = np.array(lines)[order]
    distances = distances[order]
    lines = lines[:14]
    distances = distances[:14]
    lines = lines[distances < 6]
    order = np.argsort(lines[:, 1])
    lines = lines[order]

    # 2)
    tan_lines = [np.tan(x) for x in lines[:, 1]]
    diffs = [tan_lines[i+1]-tan_lines[i] for i in range(len(tan_lines)-1)]
    median = np.median([i for i in diffs if 0.1 < i < 0.5])
    member = [sum([1 if abs(median-abs(i-j)) < 0.02 else 0 for i in tan_lines]) for j in tan_lines]
    lines = np.array([lines[i] for i in range(len(lines)) if member[i] != 0])
    tan_lines = np.array([tan_lines[i] for i in range(len(tan_lines)) if member[i] != 0])

    # 3)
    # TODO improve deletion of duplicate lines
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

    # 4)
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
    """Finds mode of horizontal lines, and returns filtered lines based on this value."""

    # TODO use better method than mode
    mode = sp.stats.mode(lines[:, 1], keepdims=True)[0]
    lines = lines[abs(lines[:, 1]-mode) <= 0.01]
    return mode, lines


def refilter_horizontal_lines(lines_h, lines_v, dims):
    """1) Calculates intersection points of sorted vertical and unsorted horizontal lines, applies gaussian blur.
    2) Uses houghlines to find line crossing the most intersection points.
    3) Uses this line to return new horizontal lines.
    """

    # 1)
    intersections = np.array([intersection_polar(i, j) for i in lines_v for j in lines_h])
    canvas = np.zeros((800, 1600), np.uint8)
    for i in intersections:
        if 0 < i[1] < dims[1] and 0 < i[0] < dims[0]:
            canvas[i[1], i[0]] += 50
    canvas = sp.ndimage.gaussian_filter(canvas, sigma=1)

    # 2)
    cross_line = cv.HoughLines(canvas, rho=1, theta=np.pi/360, threshold=10,
                                min_theta=np.pi*4/3, max_theta=np.pi*3/2-0.1)[0][0]

    # 3)
    new_horizontal = np.array([[intersection_polar(cross_line, j)[1], np.pi/2] for j in lines_v])
    return new_horizontal


def transform(lines_v, lines_h, original):
    """Given 4 boundary lines, transforms original image to fit into 1000x1000 image."""

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
    m = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(original, m, (1000, 1000))
    for i in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
        cv.line(dst, [0, i], [1000, i], (255, 0, 0), 1, cv.LINE_AA)
        cv.line(dst, [i, 0], [i, 1000], (0, 255, 0), 1, cv.LINE_AA)
    return dst


def display_in_grid(img):
    """Given input image of a chess board, returns the chess board from above as a 1000x1000 image."""

    original = cv.imread(img)
    original, resized, test_photo = resize_and_process(original, (1600, 800))

    lines_h = hough_line_horizontal(test_photo)
    mode, lines_h = filter_horizontal_lines(lines_h)
    lines_h = new_horizontal_lines_after_rotation(lines_h, mode[0], [800, 400])
    test_photo = rotate_image(test_photo, mode[0]*180/np.pi-90)
    #resized = rotate_image(resized, mode[0]*180/np.pi-90)

    lines_v = hough_line_vertical(test_photo)
    points = intersection_of_lines(lines_v)
    point = gaussian_mode_of_points(points)
    lines_v = filter_vertical_lines(point, lines_v)

    lines_h = refilter_horizontal_lines(lines_h, lines_v, (1600, 800))
    transformed = transform(lines_v, lines_h, original)
    cv.imshow('ok', transformed)
    cv.waitKey(0)
    return transformed


def photos():
    st = time.time()
    display_in_grid('chessboard_photos/random_pictures/chessboard3.jpg')
    display_in_grid('chessboard_photos/random_pictures/chessboard4.jpg')
    display_in_grid('chessboard_photos/random_pictures/chessboard5.png')
    display_in_grid('chessboard_photos/random_pictures/chessboard6.jpeg')
    display_in_grid('chessboard_photos/random_pictures/opencv_frame_0.png')
    display_in_grid('chessboard_photos/random_pictures/opencv_frame_1.png')
    display_in_grid('chessboard_photos/chess_game/position1.jpg')
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
