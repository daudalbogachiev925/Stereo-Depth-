import cv2

class StereoSGBM:
    def __init__(self, min_disp=0, num_disp=96, block=5):
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
            P1=8*3*block**2, P2=32*3*block**2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32)

    def compute(self, left, right):
        return self.matcher.compute(left, right).astype("float32") / 16.0
