import numpy as np
import cv2 as cv

maxCorners = 100

def tracking(name = ""):
    t = Tracker()
    t.video(name)

class Tracker:
    def __init__(self):
        self.newFrame = None
        self.newPoints = []
        self.oldFrame = None
        self.oldPoints = []

    def video(self, name = ""):
        if name == "":
            cap = cv.VideoCapture(0)
        else :
            cap = cv.VideoCapture(name)
        if not cap.isOpened():
            print("Cannot open camera or video", name)
            exit()

        ret, self.oldFrame = cap.read()
        self.oldFrame = cv.cvtColor(self.oldFrame, cv.COLOR_BGR2GRAY)
        self.oldFrame = cv.flip(self.oldFrame, 1)
        self.oldPoints = self.detectPoints(self.oldFrame)

        lk_params = dict(winSize=(21, 21),
                         maxLevel=3,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

        colors = np.random.randint(0, 255, (maxCorners, 3))

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            self.newFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.newFrame = cv.flip(self.newFrame, 1)
            #process frame
            if len(self.oldPoints) < 10:
                self.oldPoints = self.detectPoints(self.oldFrame)
            self.newPoints, st, err = cv.calcOpticalFlowPyrLK(self.oldFrame, self.newFrame, self.oldPoints, None, **lk_params)
            good_new = self.newPoints[st == 1]
            good_old = self.oldPoints[st == 1]


            # Display the resulting frame
            self.ShowGoodFeaturesPoints(self.newFrame, good_new)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.newFrame = cv.line(self.newFrame, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 1)
                self.newFrame = cv.circle(self.newFrame, (int(a), int(b)), 3, colors[i].tolist(), -1)

            cv.imshow('frame', self.newFrame)
            cv.setMouseCallback('frame', self.mouse_click)
            if cv.waitKey(10) == ord('q'):
                break
            self.oldFrame = self.newFrame.copy()
            self.oldPoints = good_new.reshape(-1, 1, 2)


        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()


    def ShowGoodFeaturesPoints(self, img, points):
        corners = np.int0(points)
        for i in corners:
            x, y = i.ravel()
            cv.circle(img, (x, y), 3, 255, -1)


    def detectPoints(self, img):
        feature_params = dict(maxCorners=maxCorners,
                              qualityLevel=0.01,
                              minDistance=7,
                              blockSize=7)

        points = cv.goodFeaturesToTrack(img, mask=None, **feature_params)

        return points


    def mouse_click(self, event, x, y,
                    flags, param):
        return
        if event == cv.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.roi = ((0, 0), (0, 0))
            self.oldPoints = np.asarray(list())
            self.newPoints = np.asarray(list())
        elif event == cv.EVENT_MOUSEMOVE:
            if self.start[0] >= 0:
                end = (x, y)
                self.roi = (self.start, end)

        elif event == cv.EVENT_LBUTTONUP:
            self.oldPoints = np.asarray(list())
            self.newPoints = np.asarray(list())
            end = (x, y)
            self.roi = (self.start, end)
            self.start = (-1, -1)
