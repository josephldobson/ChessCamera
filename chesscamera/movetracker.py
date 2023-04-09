from linefinder import *
import chess
import chess.pgn

def create_weird_boundaries(lines_h, lines_v, orientation):
    points = np.array([[intersection_polar(i, j) for i in lines_h] for j in lines_v])
    squares = []
    if orientation == 'Left':
        for i in reversed(range(8)):
            for j in range(8):
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
    else:
        for i in range(8):
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

def best_candidate(values, board):
    moves = list(board.legal_moves)

    lst = []
    for i in moves:

        a = board.piece_map()
        a = [a[j] if j in a else 'a' for j in range(64)]


        board.push(i)
        b = board.piece_map()

        b = [b[j] if j in b else 'a' for j in range(64)]
        change = np.array([0 if a[j] == b[j] else values[j] for j in range(64)])
        val = change[change!=0].mean()
        lst.append(val)
        board.pop()

    lst = np.array(lst)
    value = np.argmax(lst)
    board.push(moves[value])
    return board, moves[value]

def test_image(orig1, orig2, board, orientation):
    h = orig1.shape[0]
    img1 = orig1[int(h * 0.3):h, :]
    img1 = cv.resize(img1, (1600, 800), interpolation=cv.INTER_AREA)
    lines_h1, lines_v1, img1 = find_lines(img1, False)

    img2 = orig2[int(h * 0.3):h, :]
    img2 = cv.resize(img2, (1600, 800), interpolation=cv.INTER_AREA)

    a = img1.astype(np.int16)
    b = img2.astype(np.int16)

    weird1 = np.subtract(a, b)/2
    weird1 = np.absolute(weird1).astype(np.uint8)

    total_change = np.sum(weird1)
    print(total_change)
    if total_change > 200000000000   :
        lines_h2, lines_v2, img2 = find_lines(img2, False)
        img2 = transform(lines_v1, lines_h1, lines_v2, lines_h2, img1)
        b = img2.astype(np.int16)
        weird1 = np.subtract(a, b) / 2
        weird1 = np.absolute(weird1).astype(np.uint8)

    weird1 = cv.blur(weird1, (15, 15))
    weird1 = cv.blur(weird1, (17, 17))
    cv.imshow('dif', weird1)

    bounds = create_weird_boundaries(lines_h1, lines_v1, orientation)

    values = give_values(bounds, weird1)

    board, move = best_candidate(values, board)
    print("----------------")
    print(board)

    #indices = [0,1,2,3,8,9]
    #[cv.polylines(img2, [bounds[i]], True, (255, 255, 255)) for i in indices]
    cv.imshow('diff', img2)
    cv.waitKey(0)
    return board

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

if __name__ == "__main__":
    board = chess.Board()
    for i in range(len(imgs2)-1):
        im1 = cv.imread(imgs2[i])
        im2 = cv.imread(imgs2[i+1])
        board = test_image(im1,im2, board, 'Right')
