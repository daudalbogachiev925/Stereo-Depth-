import cv2

class StereoPipeline:
    def __init__(self, K, baseline):
        from src.stereo_matcher import StereoSGBM
        from src.pointcloud import disparity_to_pointcloud
        self.matcher = StereoSGBM()
        self.K = K
        self.baseline = baseline
        self.disparity_to_pointcloud = disparity_to_pointcloud

    def process(self, left, right):
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disp = self.matcher.compute(gray_l, gray_r)
        pc = self.disparity_to_pointcloud(disp, self.K, self.baseline)
        return disp, pc
