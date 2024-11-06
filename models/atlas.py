import numpy as np
import math
import nibabel as nib
import os
from PrettyPrint import figures
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter
import warnings

class Atlas:
    def __init__(self, labels=4):
        self.tissues = labels  # assuming gts are labeled from 0-N where 0 is background and N is the highest label in integer steps from 1

    def fit(self, images, gts, affine):
        if len(images) != len(gts):
            raise ValueError('Images and Gts must have the same amount')
        self.imgs = images
        self.gts = gts
        self.affine = affine
        self._build_atlas()
        self._build_reference()

    def _build_atlas(self):
        stack = np.stack(self.gts, axis=-1)
        x, y, z = stack.shape[:-1]
        self.probabilistic = np.zeros((x, y, z, self.tissues), dtype=np.float64)
        for label in range(self.tissues):
            #print('processing label', label)
            for gt in self.gts:
                mask = (gt == label)
                self.probabilistic[..., label] += mask
        # build probabilistic atlas
        axissum = np.sum(self.probabilistic, axis=-1)
        for i in range (self.tissues): # get normalized probabs, ensures that the sum of probabs along axis 4 is 1
            self.probabilistic[..., i] /= axissum
        # build anatomical atlas
        self.anatomical = np.argmax(self.probabilistic, axis=-1)
        self.mask = self.anatomical != 0

    def _build_reference(self):
        self.reference_all = np.mean(np.stack(self.imgs, axis=-1), axis=-1)
        for i in range(len(self.imgs)):
            mask = self.gts[i] == 0
            self.imgs[i][mask] = 0
        stack = np.stack(self.imgs, axis=-1)
        self.reference_brain = np.mean(stack, axis=-1) # get mean GLs along last axis
        #self.reference[np.invert(self.mask)] = 0

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        nib.save(nib.Nifti1Image(self.reference_brain, self.affine), os.path.join(path,'atlas_reference_brain.nii'))
        nib.save(nib.Nifti1Image(self.reference_all, self.affine), os.path.join(path, 'atlas_reference_all.nii'))
        nib.save(nib.Nifti1Image(self.anatomical.astype(np.float64), self.affine), os.path.join(path, 'atlas_anatomical.nii'))
        nib.save(nib.Nifti1Image(self.mask.astype(np.float64), self.affine), os.path.join(path, 'atlas_mask.nii'))
        labels = {'BG': 0, 'CSF': 1, 'GM':2, 'WM':3}
        for label in labels.keys():
            nib.save(nib.Nifti1Image(self.probabilistic[:,:,:, labels[label]], self.affine), os.path.join(path, 'atlas_probabilistic_'+label+'.nii'))

    def load(self, path):
        names = ['atlas_probabilistic_BG.nii', 'atlas_probabilistic_CSF.nii', 'atlas_probabilistic_GM.nii', 'atlas_probabilistic_WM.nii']
        maps = []
        for i in range(self.tissues):
            img = nib.load(os.path.join(path, names[i]))
            maps.append(img.get_fdata())
        self.probabilistic = np.stack(maps, axis=-1)
        '''
        img = nib.load(os.path.join(path, 'atlas_anatomical.nii'))
        self.anatomical = img.get_fdata()
        self.affine = img._affine
        self.mask = self.anatomical != 0
        '''


    def segment(self):
        return self.anatomical

    def soft_segment(self):
        return self.probabilistic[..., 1:]
