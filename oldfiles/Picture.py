def find_colours(im, lines_h, lines_v):
    # apply mask to image
    point = np.array([intersection_polar(lines_h[i[0]], lines_v[i[1]]) for i in [[0, 8], [8, 8], [8, 0], [0, 0]]])
    point = np.array([[point[0][0]-80, point[0][1]-80], [point[1][0]-200, point[1][1]],
                      [point[2][0]+200, point[2][1]], [point[3][0]+80, point[3][1]-80]])
    mask = np.zeros(im.shape, dtype=np.uint8)
    cv.fillPoly(mask, pts=[point], color=(255, 255, 255))
    im = cv.bitwise_and(im, mask)

    # Find 4 colours
    clusters = 4
    small = cv.resize(im, (0, 0), fx=0.05, fy=0.05)
    mask_s = cv.resize(mask, (0, 0), fx=0.05, fy=0.05)
    shap = small.shape
    mask_s = np.array([[mask_s[i][j][0] for i in range(shap[0])] for j in range(shap[1])])
    ar = small.reshape(np.product(shap[:2]), shap[2]).astype(float)
    mask_s = mask_s.flatten()
    ar = [ar[i] for i in range(len(ar)) if mask_s[i] == 255]
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
    c = cv.bitwise_and(c, mask)
    return c


def average_colour(im, bounds):
    lst = np.array([[0,0,0,0]])
    col = ((0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0))
    im = cv.resize(im, (0, 0), fx=0.2, fy=0.2)
    shap = im.shape

    for i in bounds:
        mask = np.zeros(shap, dtype=np.uint8)
        cv.fillPoly(mask, pts=[i], color=(255, 255, 255))
        mask_flat = mask.reshape(np.product(shap[:2]), shap[2])
        image_flat = im.reshape(np.product(shap[:2]), shap[2])

        index = np.where((mask_flat == (255, 255, 255)).all(axis=1))[0]
        image_flat = image_flat[index]
        unq, cnt = np.unique(image_flat, axis=0, return_counts=True)

        cols = [cnt[np.where((unq == i).all(axis=1))].tolist() for i in col]
        cols = np.array([i[0] if i != [] else 0 for i in cols])
        cols = np.array([cols*100/np.sum(cnt)])

        lst = np.append(lst, cols,axis = 0)
    print(len(lst))
    lst = np.delete(lst,0,0)
    print(len(lst))
    return lst