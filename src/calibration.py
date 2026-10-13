import cv2
import numpy as np
import glob

class StereoCalibrator:
    def __init__(self, pattern=(9, 6), square_size=0.025):
        self.pattern = pattern
        self.square_size = square_size

    def calibrate(self, left_imgs, right_imgs):
        objp = np.zeros((self.pattern[0]*self.pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern[0], 0:self.pattern[1]].T.reshape(-1, 2)
        objp *= self.square_size

        objpoints, imgpoints_l, imgpoints_r = [], [], []
        for lf, rf in zip(left_imgs, right_imgs):
            l = cv2.imread(lf, 0); r = cv2.imread(rf, 0)
            ok_l, cl = cv2.findChessboardCorners(l, self.pattern)
            ok_r, cr = cv2.findChessboardCorners(r, self.pattern)
            if ok_l and ok_r:
                objpoints.append(objp)
                imgpoints_l.append(cl)
                imgpoints_r.append(cr)

        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            l.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)
        return {"K1": K1, "D1": D1, "K2": K2, "D2": D2, "R": R, "T": T, "E": E, "F": F}
