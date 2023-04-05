import math
import cv2 as cv
import numpy as np
import scipy as sp


def hough_line_vertical(img):
    """Returns vertical lines in input image using the houghlines algorithm."""

    lines = cv.HoughLines(img, rho=1, theta=np.pi/360, threshold=150, srn=0, stn=0, min_theta=-1, max_theta=1)
    lines = lines[:50]
    lines = np.reshape(lines, (-1, 2))
    return lines


def hough_line_horizontal(img):
    """Returns horizontal lines in input image using the houghlines algorithm."""

    lines = cv.HoughLines(img, rho=1, theta=np.pi/360, threshold=100, min_theta=np.pi/2-0.3, max_theta=np.pi/2+0.3)
    lines = lines[:100]
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


def gaussian_mode_of_points(points, sig):
    """Given an array of 2D points, applies a Gaussian blur, and outputs the largest point."""

    # TODO finetune point selection
    canvas = np.zeros((800, 800))
    for i in points:
        x = i[0]-400
        y = i[1]+400
        if 0 < x < 800 and 0 < y < 800:
            canvas[x, y] += 1
    blurred = sp.ndimage.gaussian_filter(canvas, sigma=sig)
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


def draw_lines(Lines, col, thick, iterater, image):
    for i in iterater:
        rho = Lines[i][0]
        theta = Lines[i][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
        cv.line(image, pt1, pt2, col, thick, cv.LINE_AA)
    return image


def filter_vertical_lines(point, lines):
    """
    1) Filters the vertical lines by distance to their disappearing point.
    2) Removes lines not following pattern
    3) Uses correct lines to create a set of fine-tuned lines.
    4) Returns all 9 vertical lines from the chess board.
    """

    # 1)
    distances = np.array([distance_point_and_line(point, i) for i in lines])
    order = np.argsort(distances)
    lines = np.array(lines)[order]
    distances = distances[order]
    lines = lines[:14]
    distances = distances[:14]
    lines = lines[distances < 7]
    order = np.argsort(lines[:, 1])
    lines = lines[order]

    # 2)
    tan_lines = [np.tan(x) for x in lines[:, 1]]
    diffs = [tan_lines[i+1]-tan_lines[i] for i in range(len(tan_lines)-1)]
    median = np.median([i for i in diffs if 0.1 < i < 0.5])
    member = [sum([1 if abs(median-abs(i-j)) < 0.02 else 0 for i in tan_lines]) for j in tan_lines]
    lines = np.array([lines[i] for i in range(len(lines)) if member[i] != 0])
    tan_lines = np.array([tan_lines[i] for i in range(len(tan_lines)) if member[i] != 0])

    # Filters Vertical lines based on distance from best fit
    angle_index = np.array([round(i/median) for i in tan_lines])
    a, b = np.polyfit(angle_index, tan_lines, 1)
    x_vals = np.unique(angle_index)

    new_lines = []
    for i in x_vals:
        best_fit = np.arctan(a * i + b)
        best_ind = np.argmin([abs(best_fit - i[1]) for i in lines])
        if -5 < i < 5:
            new_lines.append(lines[best_ind])
    a, b = np.polyfit(x_vals, [np.tan(i[1]) for i in new_lines], 1)

    # Creates missing lines
    # TODO improve this bit
    point = gaussian_mode_of_points(intersection_of_lines(new_lines), 3)
    missing = np.setdiff1d(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]), x_vals)
    for i in missing:
        missing_line = create_new_line(point, np.arctan(a * i + b))
        new_lines.append(missing_line)

    lines = np.array(new_lines)[:, 1]
    order = np.argsort(np.array(lines))
    lines = np.array(new_lines)[order]

    return lines


def filter_horizontal_lines(lines):
    """Finds mode of horizontal lines, and returns filtered lines based on this value."""

    # TODO use better method than mode
    mode = sp.stats.mode(lines[:, 1], keepdims=True)[0]
    lines = lines[abs(lines[:, 1]-mode) <= 0.02]
    return mode, lines


def refilter_horizontal_lines(lines_h, lines_v, dims):
    """1) Calculates intersection points of sorted vertical and unsorted horizontal lines, applies gaussian blur.
    2) Uses hough lines to find line crossing the most intersection points.
    3) Uses this line to return new horizontal lines.
    """

    # 1)
    intersections = np.array([intersection_polar(i, j) for i in lines_v for j in lines_h])
    canvas = np.zeros((800, 1600), np.uint8)
    for i in intersections:
        if 0 < i[1] < dims[1] and 0 < i[0] < dims[0]:
            canvas[i[1], i[0]] += 200
    canvas = sp.ndimage.gaussian_filter(canvas, sigma=0.6)
    canvas = canvas * (int(255 / np.amax(canvas)))

    # 2)
    cross_lines = cv.HoughLines(canvas, rho=1, theta=np.pi/360, threshold=10,
                                min_theta=np.pi*4/3, max_theta=np.pi*3/2-0.25)
    cross_lines = np.reshape(cross_lines, (-1, 2))[:10]
    y_intersect = [1 if 640 < intersection_polar(i, [0, 0])[1] < 770 else 0 for i in cross_lines]

    # 3)
    # TODO improve line selection
    for i in range(len(y_intersect)):
        if y_intersect[i] == 1:
            cross_line = cross_lines[i]
            break
        else:
            cross_line = cross_lines[0]

    new_horizontal = np.array([[intersection_polar(cross_line, j)[1], np.pi / 2] for j in lines_v])

    return new_horizontal


def find_lines(img, show_img):
    """Given input image of a chess board, returns the chess board from above as a 1000x1000 image."""

    out = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    out = cv.blur(out, (3, 3))
    canny_image = cv.Canny(out, 70, 210)

    try:
        # Filter horizontal lines based on mode and rotate image
        lines_h = hough_line_horizontal(canny_image)
        mode, lines_h = filter_horizontal_lines(lines_h)
        lines_h = new_horizontal_lines_after_rotation(lines_h, mode[0], [800, 400])
        canny_image = rotate_image(canny_image, mode[0]*180/np.pi-90)
        resized = rotate_image(img, mode[0]*180/np.pi-90)

        # Find dissapearing point of vertical lines and filter
        lines_v = hough_line_vertical(canny_image)
        points = intersection_of_lines(lines_v)
        point = gaussian_mode_of_points(points, 1)
        lines_v = filter_vertical_lines(point, lines_v)

        # Create horizontal lines based on vertical lines
        lines_h = refilter_horizontal_lines(lines_h, lines_v, (1600, 800))
        # draw_lines(lines_h, (255,0,0), 1, range(len(lines_h)), img)
        # draw_lines(lines_v, (255, 0, 0), 1, range(len(lines_v)), img)
        # cv.imshow('a', img)
        # cv.waitKey(0)


        #lines_h = new_horizontal_lines_after_rotation(lines_h, 2*math.pi-mode[0], [800, 400])
        #lines_v = new_horizontal_lines_after_rotation(lines_v, 2*math.pi-mode[0], [800, 400])
        return lines_h, lines_v, resized
    except:
        print("Could not find lines")
        return [], [], img

imgs2 = ['ChessgameDUCC/opencv_frame_0.png',
    'ChessgameDUCC/opencv_frame_1.png',
    'ChessgameDUCC/opencv_frame_2.png',
    'ChessgameDUCC/opencv_frame_3.png',
    'ChessgameDUCC/opencv_frame_4.png',
    'ChessgameDUCC/opencv_frame_5.png',
    'ChessgameDUCC/opencv_frame_6.png',
    'ChessgameDUCC/opencv_frame_7.png',
    'ChessgameDUCC/opencv_frame_8.png',
    'ChessgameDUCC/opencv_frame_9.png',
    'ChessgameDUCC/opencv_frame_10.png',
    'ChessgameDUCC/opencv_frame_11.png',
    'ChessgameDUCC/opencv_frame_12.png',
    'ChessgameDUCC/opencv_frame_13.png',
    'ChessgameDUCC/opencv_frame_14.png',
    'ChessgameDUCC/opencv_frame_15.png',
    'ChessgameDUCC/opencv_frame_16.png',
    'ChessgameDUCC/opencv_frame_17.png',
    'ChessgameDUCC/opencv_frame_18.png',
    'ChessgameDUCC/opencv_frame_19.png',
    'ChessgameDUCC/opencv_frame_20.png',
    'ChessgameDUCC/opencv_frame_21.png',
    'ChessgameDUCC/opencv_frame_22.png',
    'ChessgameDUCC/opencv_frame_23.png',
    'ChessgameDUCC/opencv_frame_24.png',
    'ChessgameDUCC/opencv_frame_25.png',
    'ChessgameDUCC/opencv_frame_26.png',
    'ChessgameDUCC/opencv_frame_27.png',
    'ChessgameDUCC/opencv_frame_28.png',
    'ChessgameDUCC/opencv_frame_29.png',
    'ChessgameDUCC/opencv_frame_30.png',
    'ChessgameDUCC/opencv_frame_31.png',
    'ChessgameDUCC/opencv_frame_32.png',
    'ChessgameDUCC/opencv_frame_33.png',
    'ChessgameDUCC/opencv_frame_34.png',
     'ChessgameDUCC/opencv_frame_35.png',
     'ChessgameDUCC/opencv_frame_36.png',
     'ChessgameDUCC/opencv_frame_37.png',
     'ChessgameDUCC/opencv_frame_38.png',
     'ChessgameDUCC/opencv_frame_39.png',
     'ChessgameDUCC/opencv_frame_40.png',
    'ChessgameDUCC/opencv_frame_41.png',
     'ChessgameDUCC/opencv_frame_42.png',
     'ChessgameDUCC/opencv_frame_43.png',
     'ChessgameDUCC/opencv_frame_44.png',
     'ChessgameDUCC/opencv_frame_45.png',
     'ChessgameDUCC/opencv_frame_46.png',
     'ChessgameDUCC/opencv_frame_47.png',
     'ChessgameDUCC/opencv_frame_48.png',
     ]

if __name__ == "__main__":
    for i in imgs2:
        orig1 = cv.imread(i)
        h = orig1.shape[0]
        img1 = orig1[int(h * 0.3):h, :]
        img1 = cv.resize(img1, (1600, 800), interpolation=cv.INTER_AREA)
        find_lines(img1,False)
        
    #find_lines(a,b)
