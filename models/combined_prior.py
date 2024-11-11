import numpy as np
import math
import nibabel as nib
import os
from .atlas import Atlas
from .scoring import Scorer
from .tissue_model import TissueModel
from PrettyPrint import figures
from .feature import FeatureTransformer
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter
import warnings

class TMACombination():
    def __init__(self, atlas, tm):
        self.atlas = atlas
        self.tm = tm

    def segment(self, img):
        seg = self.soft_segment(img)
        mask = np.sum(seg, axis=-1) != 0
        seg = np.argmax(seg, axis=-1)
        seg[mask] += 1
        return seg

    def soft_segment(self, img):
        seg_1 = self.atlas.soft_segment(img)
        seg_2 = self.tm.soft_segment(img)
        seg = seg_1*seg_2
        return seg