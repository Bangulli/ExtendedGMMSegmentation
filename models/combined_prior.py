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
    '''
    Combines both atlas and tissue model into one prior knowledge object. Provides spatial and intensity based information
    '''
    def __init__(self, atlas, tm):
        '''
        Constructor
        :param atlas: registered atlas object
        :param tm: tissue model
        '''
        self.atlas = atlas
        self.tm = tm

    def segment(self, img):
        '''
        Segments the image
        :param img: image to segment
        :return: segmented image 3D numpy array
        '''
        seg = self.soft_segment(img)
        mask = np.sum(seg, axis=-1) != 0
        seg = np.argmax(seg, axis=-1)
        seg[mask] += 1
        return seg

    def soft_segment(self, img):
        '''
        Provides the tissue probabilities for each voxel
        :param img: image to segment
        :return: tissue probabilities for each voxel, 4D numpy array
        '''
        seg_1 = self.atlas.soft_segment(img)
        seg_2 = self.tm.soft_segment(img)
        seg = seg_1*seg_2
        return seg