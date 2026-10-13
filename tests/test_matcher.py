import numpy as np
from src.stereo_matcher import StereoSGBM

def test_disparity_shape():
    l = np.zeros((480, 640), dtype=np.uint8)
    r = np.zeros((480, 640), dtype=np.uint8)
    d = StereoSGBM().compute(l, r)
    assert d.shape == (480, 640)
