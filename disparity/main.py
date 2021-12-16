import cv2
import cv2 as cv
import numpy as np

maxCorners = 0

def disparity():
    imgA = cv.imread(cv.samples.findFile("TP/disparity/image1.jpg"))
    imgB = cv.imread(cv.samples.findFile("TP/disparity/image2_resized.jpg"))
    imgA = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
    imgB = cv.cvtColor(imgB, cv.COLOR_BGR2GRAY)
    h, w = imgA.shape
    dim = (w, h)
    ia = imgA
    ib = imgB
    #ib = cv.resize(imgB, dim, interpolation=cv.INTER_AREA)
    pa = []
    pb = []
    findMatchings(ia, ib, pa, pb)
    findMatchings(ib, ia, pb, pa)
    displayMatchings(ia, ib, pa, pb)
    iar, ibr = rectify(ia, ib, pa, pb)
    res = cv.hconcat([iar, ibr])
    cv.imshow(f"Rectified", res)
    k = cv.waitKey(0)
    disp = computeDisparity(iar, ibr)
    cv.imshow(f"Disparity", disp)
    k = cv.waitKey(0)


def findMatchings(ia, ib, pa, pb):
    feature_params = dict(maxCorners=maxCorners,
                          qualityLevel=0.05,
                          minDistance=25,
                          blockSize=7)
    lk_params = dict(winSize=(21, 21),
                     maxLevel=5,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
    tmpA = cv.goodFeaturesToTrack(ia, mask=None, **feature_params)
    tmpB, st, err = cv.calcOpticalFlowPyrLK(ia, ib, tmpA, None, **lk_params)

    if tmpB is not None:
        pB1 = tmpB[st > 0]
        pA1 = tmpA[st > 0]

        pa.extend(pA1)
        pb.extend(pB1)

def displayMatchings(ia, ib, pa, pb):

    colors = np.random.randint(0, 255, (max(maxCorners, 1000), 3))

    _,w = ia.shape
    res = cv2.hconcat([ia, ib])
    mask = np.zeros_like(res)
    for i, (new, old) in enumerate(zip(pb, pa)):
        a, b = new.ravel()
        c, d = old.ravel()
        res = cv.line(res, (int(a), int(b)), (int(c) + w, int(d)), colors[i].tolist(), 1)
        res = cv.circle(res, (int(a), int(b)), 3, colors[i].tolist(), -1)

    res = cv.add(res, mask)
    cv.imshow(f"object", res)

    k = cv.waitKey(0)

def rectify(ia, ib, pa, pb):
    pts1 = np.int32(pa)
    pts2 = np.int32(pb)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
    #pts1 = pts1[mask.ravel() == 1]
    #pts2 = pts2[mask.ravel() == 1]
    h,w = ia.shape
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (w,h))
    iar = cv.warpPerspective(ia, H1, (w,h))
    ibr = cv.warpPerspective(ib, H2, (w,h))

    return iar, ibr

def computeDisparity(iar, ibr):
    stereo = cv.StereoBM_create()
    disp = stereo.compute(iar, ibr).astype(float) / 16.
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(disp)
    coeff = 255.0 / (maxVal - minVal)
    offset = -minVal * coeff
    disp = disp * coeff + offset
    disp = np.uint8(disp)

    return disp
