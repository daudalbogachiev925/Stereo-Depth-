import cv2
import numpy as np

def disparity_to_pointcloud(disparity, K, baseline):
    h, w = disparity.shape
    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]
    points = []
    for v in range(0, h, 4):
        for u in range(0, w, 4):
            d = disparity[v, u]
            if d <= 0: continue
            z = fx * baseline / d
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
    return np.array(points)
