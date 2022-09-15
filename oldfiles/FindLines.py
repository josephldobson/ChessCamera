import random
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import time
from scipy import stats


def draw_lines(Lines, col, thick, iterater,image):
    for i in iterater:
        rho = Lines[i][0][0]
        theta = Lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
        cv.line(image, pt1, pt2, col, thick, cv.LINE_AA)

def chessboardfinderL(img,maskedimg,mid):
    start = time.process_time()
    col1 = (255,0,0)
    col2 = (0,255,0)
    col3 = (0,0,255)

    #Find Lines Within Side of image

    Vert1 = cv.HoughLines(maskedimg, rho=1, theta=np.pi/360, threshold=100, srn=0, stn=0, min_theta=0, max_theta=1)
    Vert2 = cv.HoughLines(maskedimg, rho=1, theta=np.pi/360, threshold=100, srn=0, stn=0, min_theta=np.pi-1, max_theta=np.pi)
    Vert = np.concatenate((Vert1,Vert2))
    Hori = cv.HoughLines(maskedimg, rho=1, theta=np.pi/360, threshold=100, srn=0, stn=0, min_theta=np.pi/2-0.3,max_theta=np.pi/2+0.3)
    Hori = Hori[:30]
    Vert = Vert[:30]

    ################################CHECK###############################################################################
    img_init = np.copy(img)
    draw_lines(Hori, col2, 1, range(len(Hori)),img_init)
    draw_lines(Vert, col3, 1, range(len(Vert)),img_init)
    cv.imshow('img_diss_vert',img_init)
    cv.waitKey(0)
    ################################CHECK###############################################################################

    #Find dissapearing point Vert
    def intersection_with_centre(line1):
        rho1, theta1 = line1[0]
        if np.cos(theta1)*(mid[2]-mid[0]) == (mid[1]-mid[3])*np.sin(theta1):
            return [-1000,10000]
        A = np.array([[np.cos(theta1), np.sin(theta1)] , [mid[1]-mid[3], mid[2]-mid[0]]])
        b = np.array([[rho1], [mid[2]*mid[1] - mid[0]*mid[3]]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]

    ROIVERT = [0,1100,-400,400]

    VertInter = map(intersection_with_centre,Vert)
    VertInter = np.array([i for i in VertInter if i[0]>ROIVERT[0] and i[0]<ROIVERT[1] and i[1]>ROIVERT[2] and i[1]<ROIVERT[3]])
    VertInter = np.array([[i[0]+l,i[1]+m] for i in VertInter for l in range(-8,9) for m in range(-8,9)])

    MODVERT = sp.stats.mode(np.array(VertInter))
    MODVERT = MODVERT[0][0]



    print('Finished Filtering Vertical lines using :' + str(MODVERT) + str(time.process_time() - start))


    #Filter vertical lines based on dissapearing points
    dissRAD = 10
    VertSort = []
    for i in range(0, len(Vert)):
        rho = Vert[i][0][0]
        theta = Vert[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        c = -a * a * rho - b * b * rho
        m = MODVERT[0]
        n = MODVERT[1]
        d = abs(a * m + b * n + c) / ((a ** 2 + b ** 2) ** 0.5)
        if d < dissRAD:
            VertSort.append(Vert[i])
    VertSort = sorted(VertSort, key=lambda x: x[0][0])

    print('Finished Filtering Vertical lines using :'+str(MODVERT)+str(time.process_time() - start))

    #Filter Horizontal Lines based on mode
    Horimode = stats.mode(Hori[:,0,1])[0][0]
    HoriSort = [i for i in Hori if i[0][1] == Horimode]
    HoriSort = sorted(HoriSort, key=lambda x: x[0][0])

    ################################CHECK###############################################################################
    print('Finished Filtering Horizontal lines in:'+str(time.process_time() - start))
    img_hori = np.copy(img)
    draw_lines(HoriSort, col2, 1, range(len(HoriSort)),img_hori)
    draw_lines(VertSort, col3, 1, range(len(VertSort)),img_hori)
    cv.imshow('img_diss_vert',img_hori)
    cv.waitKey(0)
    ################################CHECK###############################################################################

    #Intersection of vertical and horizontal
    def intersection_polar(line1, line2):
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]
    Inte = [[intersection_polar(i, j) for i in VertSort] for j in HoriSort]

    #Disapearing point of crossing lines

    def liner(p1,p2):
        return([p1[1]-p2[1],p2[0]-p1[0],p1[0]*p2[1]-p1[1]*p2[0]])

    def distance(p1,p2):
        return(((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5)

    CrossLines = []
    for i in range(len(HoriSort)):
        for j in range(len(VertSort)):
            for p in range(8):
                for k in range(8):
                    if i+p<len(HoriSort) and j+k<len(VertSort):
                        pt1 = Inte[i][j]
                        pt2 = Inte[i+p][j+k]
                        distL = distance(pt1,pt2)
                        L = liner(pt1,pt2)
                        if -L[0]<1*L[1]:
                            CrossLines.append([L,pt1,pt2,[i,j],[i+p,j+k],int(distL)])
    print('Calculated Cross Lines in: '+str(time.process_time() - start))

    crossinte = []
    crossintearea = []
    amain = np.cos(Horimode)
    bmain = np.sin(Horimode)

    for i in CrossLines:
        for j in range(-20,21,1):
            cmain = amain * MODVERT[0] + bmain * (MODVERT[1]+j)
            if i[0][0]*amain != i[0][1]*bmain and i[5]<130 and i[5]>80:
                A = np.array([[i[0][0], i[0][1]], [amain, bmain]])
                b = np.array([[-i[0][2]], [cmain]])
                x0, y0 = np.linalg.solve(A, b)
                x0, y0 = int(np.round(x0)), int(np.round(y0))
                #crossinte.append([x0, y0])
                crossintearea.append([x0,y0])
    print('Calculated Cross Line Intersections in: '+str(time.process_time() - start))

    crossintearea = np.array([i for i in crossintearea if i[0]<0 and i[0]>-2000])
    Ran = np.amax(crossintearea, 0) - np.amin(crossintearea, 0)
    Bins = np.histogram2d(crossintearea[:, 0], crossintearea[:, 1], bins=[100, 1])
    binsize = int(Ran[0] / 100)
    Area = np.array([sum([Bins[0][i] for i in range(j,j+5)]) for j in range(len(Bins[0])-5)])
    Areamax = np.argmax(Area)
    Areax = -2000 + Areamax*binsize
    Areay = Areax + 5*binsize

    crossinte = np.array([i for i in crossintearea if i[0]<Areay and i[0]>Areax])

    crossintemod = np.array([[i[0] + l, i[1] + m] for i in crossinte for l in range(-5, 6) for m in range(-5, 6)])

    crossintenp = np.array(crossintemod)
    MODVERT = stats.mode(crossintenp)
    MODVERT = MODVERT[0][0]
    Centx = MODVERT[0]
    Centy = MODVERT[1]
    dissRAD = 2
    print('Finished finding crossing point: ' + str([Centx,Centy])+' in: '+str(time.process_time() - start))

    #Filter cross lines based on dissapearing points


    final_lines = []
    final_points = []
    final_indices = []

    for i in CrossLines:
        a = i[0][0]
        b = i[0][1]
        c = i[0][2]
        d = abs(a * Centx + b * Centy + c) / ((a**2 + b**2) ** 0.5)
        if d < dissRAD:
            final_lines.append(i)
            final_indices.extend([[i[1][0],i[1][1],i[3][0],i[3][1]], [i[2][0],i[2][1],i[4][0],i[4][1]]])
            final_points.extend([i[1],i[2]])



    final_indices = np.array(final_indices)
    final_indices = np.unique(final_indices, axis=0)
    final_points = np.copy(final_indices)
    final_points = np.delete(final_points,[2,3],1)


    #Create Adjacency Matrix
    adj_mat = np.zeros((len(final_points),(len(final_points))))

    #      Diaganol adjancency
    for i in final_lines:
        pt1 = i[1]
        pt2 = i[2]
        pt1_ind = np.where(np.all(final_points == pt1, axis=1))[0]
        pt2_ind = np.where(np.all(final_points == pt2, axis=1))[0]
        for j in pt1_ind:
            for k in pt2_ind:
                adj_mat[j, k] = 2
                adj_mat[k, j] = 2

    #      Vertical and Horizontal adjancency
    for i in range(len(final_points)):
        for j in range(len(final_points)):
            if final_indices[i][2] == final_indices[j][2]:
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
            if final_indices[i][3] == final_indices[j][3]:
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
    adj_mat -= np.identity(len(final_points))
    sum_adjmat = adj_mat.sum(axis=1)

    FinalSet = np.array([final_indices[i] for i in range(len(final_indices)) if sum_adjmat[i]>1])
    FinalVert = FinalSet[:,3]
    FinalHori = FinalSet[:,2]
    FVU,FVUcount = np.unique(FinalVert, return_counts=True)
    FHU,FHUcount = np.unique(FinalHori, return_counts=True)

    ################################CHECK###############################################################################
    print('Finished filtering crossing lines in: '+str(time.process_time() - start))
    img_cross = np.copy(img)
    for i in final_lines:
        i = i[0]
        if i[1] != 0:
            a = i[0]
            b = i[1]
            c = i[2]
            pt1 = (-3000, int((-c - a * -3000) / b))
            pt2 = (3000, int((-c - a * 3000) / b))
            cv.line(img_cross, pt1, pt2, col2, 1, cv.LINE_AA)
    for p in final_points:
        cv.circle(img_cross,p,3,col1,1)
    draw_lines(HoriSort, col1, 1, FinalHori, img_cross)
    draw_lines(VertSort, col3, 1, FinalVert, img_cross)
    cv.imshow('img_cross',img_cross)
    cv.waitKey(0)
    ################################CHECK###############################################################################


    #Filter Duplicates Vertical

    HoriMax = HoriSort[np.amax(FinalHori)]
    VertMin = VertSort[np.amin(FinalVert)]
    FinalVertSort = []

    oldinter = intersection_polar(HoriMax,VertMin)
    i = np.amin(FinalVert)
    iind = 0
    for ind in range(len(FVU)):
        newinter = intersection_polar(HoriMax,VertSort[FVU[ind]])

        if abs(oldinter[0]-newinter[0])<20 and FVUcount[ind]>=FVUcount[iind]:
            i = FVU[ind]
            iind = ind
        if abs(oldinter[0] - newinter[0]) > 20:
            FinalVertSort.extend([i])
            oldinter = newinter
            i = FVU[ind]
            iind = ind
    FinalVertSort.extend([i])

    #Filter Duplicates Horizontal

    HoriMin = HoriSort[np.amin(FinalHori)]
    VertMax = VertSort[np.amax(FinalVert)]
    FinalHoriSort = []

    oldinter = intersection_polar(VertMax,HoriMin)
    i = np.amin(FinalHori)
    iind = 0
    for ind in range(len(FHU)):
        newinter = intersection_polar(VertMax,HoriSort[FHU[ind]])

        if abs(oldinter[1]-newinter[1])<20 and FHUcount[ind]>=FHUcount[iind]:
            i = FHU[ind]
            iind = ind
        if abs(oldinter[1] - newinter[1]) > 20:
            FinalHoriSort.extend([i])
            oldinter = newinter
            i = FHU[ind]
            iind = ind
    FinalHoriSort.extend([i])

    Horizontal = [HoriSort[i] for i in FinalHoriSort]
    Vertical = [VertSort[i] for i in FinalVertSort]


    ################################CHECK###############################################################################
    print('Finished filtering final lines: '+str(time.process_time() - start))
    img_final = np.copy(img)
    draw_lines(Horizontal, col1, 1, range(len(Horizontal)), img_final)
    draw_lines(Vertical, col3, 1, range(len(Vertical)), img_final)
    cv.imshow('img_cross',img_final)
    cv.waitKey(0)
    ################################CHECK###############################################################################


    #Pick 4 points and location
    top = Horizontal[0]
    bot = Horizontal[len(Horizontal)-1]
    lef = Vertical[0]
    rig = Vertical[len(Vertical)-1]
    pt1 = intersection_polar(top, lef)
    pt2 = intersection_polar(top, rig)
    pt3 = intersection_polar(bot, lef)
    pt4 = intersection_polar(bot, rig)
    height = len(Horizontal)-1
    width = len(Vertical)-1


    def transform(img, pts, w, h):
        pts1 = np.float32([pts[0], pts[1], pts[2], pts[3]])
        pts2 = np.float32([[500-w*100, 900-h*100], [500, 900-h*100], [200, 900], [500, 900]])
        M = cv.getPerspectiveTransform(pts1, pts2)
        dst = cv.warpPerspective(img, M, (1000,1000))
        for i in [100,200,300,400,500,600,700,800,900]:
            cv.line(dst,[0,i],[500,i],col1,1)
        for i in [100,200,300,400]:
            cv.line(dst,[i,0],[i,1000],col1,1)
        return (dst)


    cool = transform(img, [pt1,pt2,pt3,pt4], width, height)
    #Draw Points

    for i in FinalSet:
        pt =[i[0],i[1]]
        cv.circle(img,pt,4,col2)


    draw_lines(Horizontal,col1,1,range(len(Horizontal)),img)

    draw_lines(Vertical, col2, 1, range(len(Vertical)),img)


    print('Finished drawing lines in: '+str(time.process_time() - start))
    return(img,cool)

def chessboardfinderR(img,maskedimg,mid):
    start = time.process_time()
    col1 = (255,0,0)
    col2 = (0,255,0)
    col3 = (0,0,255)

    #Find Lines Within Side of image

    Vert1 = cv.HoughLines(maskedimg, rho=1, theta=np.pi/360, threshold=100, srn=0, stn=0, min_theta=0, max_theta=1)
    Vert2 = cv.HoughLines(maskedimg, rho=1, theta=np.pi/360, threshold=100, srn=0, stn=0, min_theta=np.pi-1, max_theta=np.pi)
    Vert = np.concatenate((Vert1,Vert2))
    Hori = cv.HoughLines(maskedimg, rho=1, theta=np.pi/360, threshold=100, srn=0, stn=0, min_theta=np.pi/2-0.3,max_theta=np.pi/2+0.3)
    Hori = Hori[:30]
    Vert = Vert[:30]

    ################################CHECK###############################################################################
    img_init = np.copy(img)
    draw_lines(Hori, col2, 1, range(len(Hori)),img_init)
    draw_lines(Vert, col3, 1, range(len(Vert)),img_init)
    cv.imshow('img_diss_vert',img_init)
    cv.waitKey(0)
    ################################CHECK###############################################################################

    #Find dissapearing point Vert
    def intersection_with_centre(line1):
        rho1, theta1 = line1[0]
        if np.cos(theta1)*(mid[2]-mid[0]) == (mid[1]-mid[3])*np.sin(theta1):
            return [-1000,10000]
        A = np.array([[np.cos(theta1), np.sin(theta1)] , [mid[1]-mid[3], mid[2]-mid[0]]])
        b = np.array([[rho1], [mid[2]*mid[1] - mid[0]*mid[3]]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]

    ROIVERT = [0,1100,-400,400]

    VertInter = map(intersection_with_centre,Vert)
    VertInter = np.array([i for i in VertInter if i[0]>ROIVERT[0] and i[0]<ROIVERT[1] and i[1]>ROIVERT[2] and i[1]<ROIVERT[3]])
    VertInter = np.array([[i[0]+l,i[1]+m] for i in VertInter for l in range(-8,9) for m in range(-8,9)])

    MODVERT = stats.mode(np.array(VertInter))
    MODVERT = MODVERT[0][0]



    print('Finished Filtering Vertical lines using :' + str(MODVERT) + str(time.process_time() - start))


    #Filter vertical lines based on dissapearing points
    dissRAD = 10
    VertSort = []
    for i in range(0, len(Vert)):
        rho = Vert[i][0][0]
        theta = Vert[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        c = -a * a * rho - b * b * rho
        m = MODVERT[0]
        n = MODVERT[1]
        d = abs(a * m + b * n + c) / ((a ** 2 + b ** 2) ** 0.5)
        if d < dissRAD:
            VertSort.append(Vert[i])
    VertSort = sorted(VertSort, key=lambda x: x[0][0])

    #Filter Horizontal Lines based on mode
    Horimode = stats.mode(Hori[:,0,1])[0][0]
    HoriSort = [i for i in Hori if i[0][1] == Horimode]
    HoriSort = sorted(HoriSort, key=lambda x: x[0][0])

    ################################CHECK###############################################################################
    print('Finished Filtering Horizontal lines in:'+str(time.process_time() - start))
    img_hori = np.copy(img)
    draw_lines(HoriSort, col2, 1, range(len(HoriSort)),img_hori)
    draw_lines(VertSort, col3, 1, range(len(VertSort)),img_hori)
    cv.imshow('img_diss_vert',img_hori)
    cv.waitKey(0)
    ################################CHECK###############################################################################

    #Intersection of vertical and horizontal
    def intersection_polar(line1, line2):
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]
    Inte = [[intersection_polar(i, j) for i in VertSort] for j in HoriSort]

    #Disapearing point of crossing lines

    def liner(p1,p2):
        return([p1[1]-p2[1],p2[0]-p1[0],p1[0]*p2[1]-p1[1]*p2[0]])

    def distance(p1,p2):
        return(((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5)

    CrossLines = []
    for i in range(len(HoriSort)):
        for j in reversed(range(len(VertSort))):
            for p in range(8):
                for k in range(8):
                    if i+p<len(HoriSort) and j-k>0:
                        pt1 = Inte[i][j]
                        pt2 = Inte[i+p][j-k]
                        distL = distance(pt1,pt2)
                        L = liner(pt1,pt2)
                        if -L[0]<-1*L[1] and -L[0]>-0.15*L[1] and distL>60:
                            CrossLines.append([L,pt1,pt2,[i,j],[i+p,j-k],int(distL)])
    print('Calculated Cross Lines in: '+str(time.process_time() - start))

    crossinte = []
    crossintearea = []
    amain = np.cos(Horimode)
    bmain = np.sin(Horimode)

    for i in CrossLines:
        for j in range(-20,21,1):
            cmain = amain * MODVERT[0] + bmain * (MODVERT[1]+j)
            if i[0][0]*amain != i[0][1]*bmain:
                A = np.array([[i[0][0], i[0][1]], [amain, bmain]])
                b = np.array([[-i[0][2]], [cmain]])
                x0, y0 = np.linalg.solve(A, b)
                x0, y0 = int(np.round(x0)), int(np.round(y0))
                crossintearea.append([x0,y0])
    print('Calculated Cross Line Intersections in: '+str(time.process_time() - start))

    crossintearea = np.array([i for i in crossintearea if i[0]<2000+1920 and i[0]>1920])

    Ran = np.amax(crossintearea, 0) - np.amin(crossintearea, 0)
    Bins = np.histogram2d(crossintearea[:, 0], crossintearea[:, 1], bins=[30, 1])
    binsize = int(Ran[0] / 30)
    print(Ran)
    Area = np.array([sum([Bins[0][i] for i in range(j,j+5)]) for j in range(len(Bins[0])-5)])
    Areamax = np.argmax(Area)
    Areax = 1920 + Areamax*binsize
    Areay = Areax + 5*binsize
    print(Areax,Areay)
    crossinte = np.array([i for i in crossintearea if i[0]<Areay and i[0]>Areax])

    crossintemod = np.array([[i[0] + l, i[1] + m] for i in crossinte for l in range(-4, 5) for m in range(-4, 5)])

    crossintenp = np.array(crossintemod)
    MODVERT = stats.mode(crossintenp)
    MODVERT = MODVERT[0][0]
    Centx = MODVERT[0]
    Centy = MODVERT[1]
    dissRAD = 2
    print('Finished finding crossing point: ' + str([Centx,Centy])+' in: '+str(time.process_time() - start))

    #Filter cross lines based on dissapearing points


    final_lines = []
    final_points = []
    final_indices = []

    for i in CrossLines:
        a = i[0][0]
        b = i[0][1]
        c = i[0][2]
        d = abs(a * Centx + b * Centy + c) / ((a**2 + b**2) ** 0.5)
        if d < dissRAD:
            final_lines.append(i)
            final_indices.extend([[i[1][0],i[1][1],i[3][0],i[3][1]], [i[2][0],i[2][1],i[4][0],i[4][1]]])
            final_points.extend([i[1],i[2]])



    final_indices = np.array(final_indices)
    final_indices = np.unique(final_indices, axis=0)
    final_points = np.copy(final_indices)
    final_points = np.delete(final_points,[2,3],1)


    #Create Adjacency Matrix
    adj_mat = np.zeros((len(final_points),(len(final_points))))

    #      Diaganol adjancency
    for i in final_lines:
        pt1 = i[1]
        pt2 = i[2]
        pt1_ind = np.where(np.all(final_points == pt1, axis=1))[0]
        pt2_ind = np.where(np.all(final_points == pt2, axis=1))[0]
        for j in pt1_ind:
            for k in pt2_ind:
                adj_mat[j, k] = 2
                adj_mat[k, j] = 2

    #      Vertical and Horizontal adjancency
    for i in range(len(final_points)):
        for j in range(len(final_points)):
            if final_indices[i][2] == final_indices[j][2]:
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
            if final_indices[i][3] == final_indices[j][3]:
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
    adj_mat -= np.identity(len(final_points))
    sum_adjmat = adj_mat.sum(axis=1)

    FinalSet = np.array([final_indices[i] for i in range(len(final_indices)) if sum_adjmat[i]>1])
    FinalVert = FinalSet[:,3]
    FinalHori = FinalSet[:,2]
    FVU,FVUcount = np.unique(FinalVert, return_counts=True)
    FHU,FHUcount = np.unique(FinalHori, return_counts=True)

    ################################CHECK###############################################################################
    print('Finished filtering crossing lines in: '+str(time.process_time() - start))
    img_cross = np.copy(img)
    for i in final_lines:
        i = i[0]
        if i[1] != 0:
            a = i[0]
            b = i[1]
            c = i[2]
            pt1 = (-3000, int((-c - a * -3000) / b))
            pt2 = (3000, int((-c - a * 3000) / b))
            cv.line(img_cross, pt1, pt2, col2, 1, cv.LINE_AA)
    for p in final_points:
        cv.circle(img_cross,p,3,col1,1)
    draw_lines(HoriSort, col1, 1, FinalHori, img_cross)
    draw_lines(VertSort, col3, 1, FinalVert, img_cross)
    cv.imshow('img_cross',img_cross)
    cv.waitKey(0)
    ################################CHECK###############################################################################


    #Filter Duplicates Vertical

    HoriMax = HoriSort[np.amax(FinalHori)]
    VertMin = VertSort[np.amin(FinalVert)]
    FinalVertSort = []

    oldinter = intersection_polar(HoriMax,VertMin)
    i = np.amin(FinalVert)
    iind = 0
    for ind in range(len(FVU)):
        newinter = intersection_polar(HoriMax,VertSort[FVU[ind]])

        if abs(oldinter[0]-newinter[0])<20 and FVUcount[ind]>=FVUcount[iind]:
            i = FVU[ind]
            iind = ind
        if abs(oldinter[0] - newinter[0]) > 20:
            FinalVertSort.extend([i])
            oldinter = newinter
            i = FVU[ind]
            iind = ind
    FinalVertSort.extend([i])

    #Filter Duplicates Horizontal

    HoriMin = HoriSort[np.amin(FinalHori)]
    VertMax = VertSort[np.amax(FinalVert)]
    FinalHoriSort = []

    oldinter = intersection_polar(VertMax,HoriMin)
    i = np.amin(FinalHori)
    iind = 0
    for ind in range(len(FHU)):
        newinter = intersection_polar(VertMax,HoriSort[FHU[ind]])

        if abs(oldinter[1]-newinter[1])<20 and FHUcount[ind]>=FHUcount[iind]:
            i = FHU[ind]
            iind = ind
        if abs(oldinter[1] - newinter[1]) > 20:
            FinalHoriSort.extend([i])
            oldinter = newinter
            i = FHU[ind]
            iind = ind
    FinalHoriSort.extend([i])

    Horizontal = [HoriSort[i] for i in FinalHoriSort]
    Vertical = [VertSort[i] for i in FinalVertSort]


    ################################CHECK###############################################################################
    print('Finished filtering final lines: '+str(time.process_time() - start))
    img_final = np.copy(img)
    draw_lines(Horizontal, col1, 1, range(len(Horizontal)), img_final)
    draw_lines(Vertical, col3, 1, range(len(Vertical)), img_final)
    cv.imshow('img_cross',img_final)
    cv.waitKey(0)
    ################################CHECK###############################################################################


    #Pick 4 points and location
    top = Horizontal[0]
    bot = Horizontal[len(Horizontal)-1]
    rig = Vertical[len(Vertical)-2]
    pt1 = intersection_with_centre(top)
    pt2 = intersection_polar(top, rig)
    pt3 = intersection_with_centre(bot)
    pt4 = intersection_polar(bot, rig)
    height = len(Horizontal)-1
    width = len(Vertical)-1
    width = 3

    def transform(img, pts, w, h):
        pts1 = np.float32([pts[0], pts[1], pts[2], pts[3]])
        pts2 = np.float32([[500, 900-h*100], [500+w*100, 900-h*100], [500, 900], [500+w*100, 900]])
        M = cv.getPerspectiveTransform(pts1, pts2)
        dst = cv.warpPerspective(img, M, (1000,1000))
        for i in [100,200,300,400,500,600,700,800,900]:
            cv.line(dst,[500,i],[1000,i],col1,1)
        for i in [500,600,700,800,900]:
            cv.line(dst,[i,0],[i,1000],col1,1)
        return (dst)


    cool = transform(img, [pt1,pt2,pt3,pt4], width, height)
    #Draw Points

    for i in FinalSet:
        pt =[i[0],i[1]]
        cv.circle(img,pt,4,col2)


    draw_lines(Horizontal,col1,1,range(len(Horizontal)),img)

    draw_lines(Vertical, col2, 1, range(len(Vertical)),img)


    print('Finished drawing lines in: '+str(time.process_time() - start))
    return(img,cool)

def find_lines(img):

    start = time.process_time()
    #resize
    img = cv.resize(img, (1920,960), interpolation=cv.INTER_AREA)
    h = img.shape[0]
    w = img.shape[1]
    img = img[int(h*0.3):h, :]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.uint8(img)
    h = int(h*0.7)

    #blur
    blur = cv.GaussianBlur(gray,(5,5),0)
    blur = cv.GaussianBlur(blur,(5,5),0)

    #canny
    canny = cv.Canny(blur, 50,100)

    #find centre line, create masks
    linesP = cv.HoughLinesP(canny, rho=2, theta=np.pi/360, threshold=180, minLineLength = 400, maxLineGap = 400)
    mid = [int(w/2),0,int(w/2),h]
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if abs((l[1]-l[3])/(l[0]-l[2])) > 0.9 and l[2] > w/2-50 and l[2] < w/2+50:
                if l[1]>l[3]: mid = l
                else: mid = [l[2],l[3],l[0],l[1]]
                break
    maskL = np.full((h, w), 0, dtype=np.uint8)
    maskR = np.full((h, w), 0, dtype=np.uint8)
    contoursL = np.array([[0,0], [mid[2],mid[3]], [mid[0],mid[1]], [0,h]])
    contoursR = np.array([[mid[2],mid[3]], [w, 0], [w,h], [mid[0],mid[1]]])
    cv.fillPoly(maskL, pts=[contoursL], color=(255,255,255))
    cv.fillPoly(maskR, pts=[contoursR], color=(255, 255, 255))

    Right = cv.bitwise_and(canny, canny, mask=maskR)
    Left = cv.bitwise_and(canny, canny, mask=maskL)

    after, transformed_left = chessboardfinderL(img,Left,mid)
    img, transformed_left = chessboardfinderR(img,Right,mid)
    final_mask = np.full((1000,1000))
    cv.fillPoly(final_mask, pts=[contoursL], color=(255, 255, 255))


    return(cool)