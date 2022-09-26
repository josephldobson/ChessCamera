from linefinder import *


def create_enlarged_boundaries(lines_h, lines_v):
    points = np.array([[intersection_polar(i, j) for i in lines_v] for j in lines_h])
    squares = []
    for i in reversed(range(8)):
        for j in reversed(range(8)):
            a = points[i][j+1]
            b = points[i+1][j+1]
            c = points[i+1][j]
            d = points[i][j]
            height = int((c[0]-b[0])*0.5+20)
            if a[0] > b[0]:
                e = [a[0], a[1]-height]
                f = [b[0], b[1]-height]
                g = b
            else:
                e = [a[0], a[1]-height]
                f = a
                g = b
            if c[0] > d[0]:
                m = c
                n = [c[0], c[1]-height]
                q = [d[0], d[1]-height]
            else:
                m = c
                n = d
                q = [d[0], d[1]-height]
            squares.append([e, f, g, m, n, q])
    squares = np.array(squares)
    return squares


def create_boundaries(lines_h, lines_v):
    points = np.array([[intersection_polar(i, j) for i in lines_v] for j in lines_h])
    squares = []
    for i in reversed(range(8)):
        for j in reversed(range(8)):
            a = points[i][j+1]
            b = points[i+1][j+1]
            c = points[i+1][j]
            d = points[i][j]
            squares.append([a, b, c, d])

    squares = np.array(squares)
    return squares


def create_weird_boundaries(lines_h, lines_v):
    points = np.array([[intersection_polar(i, j) for i in lines_v] for j in lines_h])
    squares = []
    for i in reversed(range(8)):
        for j in reversed(range(8)):
            a = points[i][j+1]
            b = points[i+1][j+1]
            c = points[i+1][j]
            d = points[i][j]
            yes = np.array([a, b, c, d])
            centre = np.mean(yes,axis=0)
            centre = centre.astype(np.int32)
            centre = centre.tolist()
            a = centre + np.array([-40,10])
            b = centre + np.array([40,10])
            c = centre + np.array([20,-100])
            d = centre + np.array([-20,-100])
            squares.append([a, b, c, d])

    squares = np.array(squares)
    return squares


def transform(lines_v1, lines_h1, lines_v2, lines_h2, original):
    """Given 4 boundary lines, transforms original image to fit into 1000x1000 image."""

    left_line1 = lines_v1[len(lines_v1)-1]
    right_line1 = lines_v1[0]
    top_line1 = lines_h1[0]
    bottom_line1 = lines_h1[len(lines_h1)-1]

    left_line2 = lines_v2[len(lines_v2)-1]
    right_line2 = lines_v2[0]
    top_line2 = lines_h2[0]
    bottom_line2 = lines_h2[len(lines_h2)-1]


    tl1 = intersection_polar(left_line1, top_line1)
    tr1 = intersection_polar(right_line1, top_line1)
    bl1 = intersection_polar(left_line1, bottom_line1)
    br1 = intersection_polar(right_line1, bottom_line1)

    tl2 = intersection_polar(left_line2, top_line2)
    tr2 = intersection_polar(right_line2, top_line2)
    bl2 = intersection_polar(left_line2, bottom_line2)
    br2 = intersection_polar(right_line2, bottom_line2)

    pts1 = np.float32([tl1, tr1, bl1, br1])
    pts2 = np.float32([tl2, tr2, bl2, br2])
    m = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(original, m, (1600, 800))
    return dst


def find_colours_2(im):

    # Find 4 colours
    clusters = 4
    small = cv.resize(im, (0, 0), fx=0.05, fy=0.05)
    shap = small.shape
    ar = small.reshape(np.product(shap[:2]), shap[2]).astype(float)
    codes, dist = sp.cluster.vq.kmeans(ar, clusters)
    codes = codes[codes[:, 2].argsort()]

    # re-colour image
    shap = im.shape
    ar = im.reshape(np.product(shap[:2]), shap[2]).astype(float)
    vecs, dist = sp.cluster.vq.vq(ar, codes)
    c_new = [[0,255,0], [0,0,255], [255,255,255], [0,0,0]]
    # c_new = [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0]]
    c = ar.copy()
    for i, code in enumerate(codes):
        c[sp.r_[np.where(vecs == i)], :] = c_new[i]
    c = np.reshape(c, shap)
    c = c.astype(np.uint8)

    return c


def give_values(bounds,img):
    lst = []
    im = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
    im = np.copy(img)
    shap = im.shape
    for i in bounds:
        mask = np.zeros(shap, dtype=np.uint8)
        cv.fillPoly(mask, pts=[i], color=(255, 255, 255))
        current = cv.bitwise_and(mask, im)
        lst.append(np.sum(current))

    return(np.array(lst))



def test_image(file1, file2):
    orig1 = cv.imread(file1)
    h = orig1.shape[0]
    img1 = orig1[int(h * 0.3):h, :]
    img1 = cv.resize(img1, (1600, 800), interpolation=cv.INTER_AREA)
    lines_h1, lines_v1, img1 = find_lines(img1, False)

    orig2 = cv.imread(file2)
    img2 = orig2[int(h * 0.3):h, :]
    img2 = cv.resize(img2, (1600, 800), interpolation=cv.INTER_AREA)
    lines_h2, lines_v2, img2 = find_lines(img2, False)
    img1 = transform(lines_v1, lines_h1, lines_v2, lines_h2, img1)

    a = img1.astype(np.int16)
    b = img2.astype(np.int16)
    weird1 = np.subtract(a, b)/2
    weird1 = np.absolute(weird1).astype(np.uint8)


    bounds = create_weird_boundaries(lines_h2,lines_v2)

    values = give_values(bounds, weird1)
    indices = values.argsort()[-6:][::-1]

    gray = cv.cvtColor(weird1, cv.COLOR_BGR2GRAY)
    ret, thresh1 = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
    cv.imshow('gray',thresh1)




    [cv.polylines(weird1,[bounds[i]],True,(255,255,255)) for i in indices]
    cv.imshow('diff',weird1)
    cv.waitKey(0)
    return

imgs = ['chessboard_photos/chess_game/position1.jpg',
    'chessboard_photos/chess_game/position2.jpg',
    'chessboard_photos/chess_game/position3.jpg',
    'chessboard_photos/chess_game/position4.jpg',
    'chessboard_photos/chess_game/position5.jpg',
    'chessboard_photos/chess_game/position6.jpg',
    'chessboard_photos/chess_game/position7.jpg',
    'chessboard_photos/chess_game/position8.jpg',
    'chessboard_photos/chess_game/position9.jpg',
    'chessboard_photos/chess_game/position10.jpg',
    'chessboard_photos/chess_game/position11.jpg',
    'chessboard_photos/chess_game/position12.jpg',
    'chessboard_photos/chess_game/position13.jpg',
    'chessboard_photos/chess_game/position14.jpg',
    'chessboard_photos/chess_game/position15.jpg',
    'chessboard_photos/chess_game/position16.jpg',
    'chessboard_photos/chess_game/position17.jpg',
    'chessboard_photos/chess_game/position18.jpg',
    'chessboard_photos/chess_game/position19.jpg',
    'chessboard_photos/chess_game/position20.jpg',
    'chessboard_photos/chess_game/position21.jpg',
    'chessboard_photos/chess_game/position22.jpg',
    'chessboard_photos/chess_game/position23.jpg',
    'chessboard_photos/chess_game/position24.jpg',
    'chessboard_photos/chess_game/position25.jpg']

for i in range(len(imgs)-1):
    test_image(imgs[i],imgs[i+1])
