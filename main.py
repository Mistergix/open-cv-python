import tracking
import features.main as feat
import disparity.main as disp

if __name__ == '__main__':
    method = 1
    # 0 = FEATURE
    # 1 = DISPARITY
    # 2 = SYMMETRY
    # 3 = TRACKING

    if method == 0:
        set = 1
        feat.features(set)
    elif method == 1:
        disp.disparity()
