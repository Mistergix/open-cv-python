import tracking
import features.main as feat
import disparity.main as disp
import symmetry.main as sym
import tracking.main as track
import ocr.main as ocr

if __name__ == '__main__':
    method = 3
    # 0 = FEATURE
    # 1 = DISPARITY
    # 2 = SYMMETRY / Not done
    # 3 = TRACKING
    # 4 = OCR / Not Done

    if method == 0:
        set = 1
        feat.features(set)
    elif method == 1:
        disp.disparity()
    elif method == 2:
        sym.symetry()
    elif method == 3:
        track.tracking("")
    elif method == 4:
        ocr.main()

