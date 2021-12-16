import cv2 as cv
import sys
import numpy as np
import ffmpeg
import string
import random as rnd


def features(set=1):
    lowercase_alphabets = list(string.ascii_lowercase)

    green = (0, 255, 0)
    loweRatio = 0.77
    MIN_GOOD_MATCHES = 10

    if set == 1:
        setName = "set1"
        imageNames = ["bleach.jpg", "deathnote.jpg", "gantz.jpg", "naruto.jpg", "yakitate.jpg"]
    elif set == 2:
        setName = "set2"
        imageNames = [f"{letter}.jpg" for letter in lowercase_alphabets[:18]]
    else:
        print("pas de set")
        imageNames = []
        setName = ""

    basePath = "TP/FeaturesSets/"

    videoName = f"{setName}/video.mp4"

    pathToVideo = f"{basePath}{videoName}"

    orb = cv.ORB_create()

    cap = cv.VideoCapture(pathToVideo)

    descriptors = []
    kps = []
    images = []
    colors = []

    for name in imageNames:
        imageName = f"{setName}/{name}"
        pathToImage = f"{basePath}{imageName}"
        img = cv.imread(cv.samples.findFile(pathToImage))
        if img is None:
            sys.exit(f"Could not read the image. {pathToImage}")

        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        cv.imshow(f"object {name}", img)

        images.append(img)
        kps.append(kp)
        descriptors.append(des)
        colors.append((rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        kpVideo = orb.detect(frame, None)
        kpVideo, desVideo = orb.compute(frame, kpVideo)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        bf.add(descriptors)
        bf.train()
        for i in range(len(images)):
            desImage = descriptors[i]
            kpImage = kps[i]
            image = images[i]
            matches = bf.knnMatch(desImage, desVideo, k=2)
            good = []
            for m, n in matches:
                if m.distance < loweRatio * n.distance:
                    good.append(m)
            if len(good) > MIN_GOOD_MATCHES:
                try:
                    src = np.float32([kpImage[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dest = np.float32([kpVideo[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv.findHomography(src, dest, cv.LMEDS, 5.0)
                    matchesMask = mask.ravel().tolist()

                    h, w, _ = image.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts, M)
                    frame = cv.polylines(frame, [np.int32(dst)], True, colors[i], 3, cv.LINE_AA)
                except Exception as e:
                    print(e)
            else:
                matchesMask = None

            print(f"Il y a {len(good)} matches for {imageNames[i]}")

        cv.imshow("scene", frame)

        if cv.waitKey(30) == ord('q'):
            break

    k = cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()
