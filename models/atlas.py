import numpy as np
import math
import nibabel as nib
import os
from PrettyPrint import figures
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter
import warnings

class Atlas:
    '''
    The atlas class. An object used for building an loading probabilistic and anatomical atlases as well as reference image and brain mask
    '''
    def __init__(self, labels=4):
        '''
        Cunstructor to instantiate the class
        :param labels: amount of lables, default is 4 for background, cerebrospinal fluid, grey matter and white matter.
        '''
        self.tissues = labels  # assuming gts are labeled from 0-N where 0 is background and N is the highest label in integer steps from 1

    def build(self, images, gts, affine):
        '''
        Builds the atlas with the passed images
        Images must be registered already to form the atlas. Recommended tool is elastix.
        :param images: list of volumetric images (=3D numpy arrays)
        :param gts: list of ground truth label images (=3D numpy arrays)
        :param affine: the affine transformation matrix of the reference image of the registration
        :return:
        '''
        if len(images) != len(gts):
            raise ValueError('Images and Gts must have the same amount')
        self.imgs = images
        self.gts = gts
        self.affine = affine
        self._build_atlas()
        self._build_reference()

    def _build_atlas(self):
        '''
        Builds the atlas internally
        probabilistic atlas is a 4D numpy array where the 4th dimension is the tissue type. Contains probability of a voxel in space belonging to the corresponding class.
        probabilities are normalized along the last axis
        anatomical atlas is a 3D numpy array with the labels of the highest probability in the probabilistic atlas in each voxel
        :return:
        '''
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
        '''
        Builds the reference image for the atlas by averaging the intensities in the registered images
        :return:
        '''
        self.reference_all = np.mean(np.stack(self.imgs, axis=-1), axis=-1)
        for i in range(len(self.imgs)):
            mask = self.gts[i] == 0
            self.imgs[i][mask] = 0
        stack = np.stack(self.imgs, axis=-1)
        self.reference_brain = np.mean(stack, axis=-1) # get mean GLs along last axis
        #self.reference[np.invert(self.mask)] = 0

    def save(self, path):
        '''
        Saves the atlas to a location, creates a directory and stores the files for each attribute of the atlas inside
        :param path: string, path to the location optionally with a new folder name
        :return:
        '''
        os.makedirs(path, exist_ok=True)
        nib.save(nib.Nifti1Image(self.reference_brain, self.affine), os.path.join(path,'atlas_reference_brain.nii'))
        nib.save(nib.Nifti1Image(self.reference_all, self.affine), os.path.join(path, 'atlas_reference_all.nii'))
        nib.save(nib.Nifti1Image(self.anatomical.astype(np.float64), self.affine), os.path.join(path, 'atlas_anatomical.nii'))
        nib.save(nib.Nifti1Image(self.mask.astype(np.float64), self.affine), os.path.join(path, 'atlas_mask.nii'))
        labels = {'BG': 0, 'CSF': 1, 'GM':2, 'WM':3}
        for label in labels.keys():
            nib.save(nib.Nifti1Image(self.probabilistic[:,:,:, labels[label]], self.affine), os.path.join(path, 'atlas_probabilistic_'+label+'.nii'))

    def load(self, path):
        '''
        Load the atlas components from a location
        :param path:
        :return:
        '''
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
        self.mask = self.anatomical != 0'''



    def segment(self):
        '''
        Segments the atlas. Basically just returns the anatomical atlas, if the atlas is registered to a new target image, the anatomical atlas becomes the images segmentation
        :return:
        '''
        return self.anatomical

    def soft_segment(self):
        '''
        Gets the probabilities of a voxel to belong to a class.
        Basically just returns the porbabilistic atlas, without using the background labels
        Used in the atlas gmm
        :return:
        '''
        return self.probabilistic[..., 1:]
