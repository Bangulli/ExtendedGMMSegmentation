import numpy as np

class FeatureTransformer():
    '''
    Feature Transformer object to convert image data to feature space
    '''
    def transform(self, images, mask):
        '''
        Takes the images and combines the intensities into a dataset of features
        :param images: list, list of images to include
        :param mask: nd array, mask of voxels to include
        :return: Nx2 array of features
        '''
        if not isinstance(images, list):
            raise TypeError('Images needs to be a list of images, even if its just one image used.')

        #if not images[0].shape == images[1].shape:
            #raise ValueError('Images must be of equal size')
        self.data = images
        self.shape = images[0].shape
        self.pixels = np.asarray(np.argwhere(mask))
        #print(self.pixels)
        self.features = np.zeros((self.pixels.shape[0], len(self.data)))
        for i in range(len(self.data)):
            self.features[:, i] = self.data[i][tuple(self.pixels.T)]
        return self.features

    def retransform(self, images):
        '''
        Takes the images and combines the intensities into a dataset of features
        Uses the internally stored coordinates for the feature transformation.
        :param images: list, list of images to include
        :return: Nx2 array of features
        '''
        if not isinstance(images, list):
            raise TypeError('Images needs to be a list of images, even if its just one image used.')

        #if not images[0].shape == images[1].shape:
            #raise ValueError('Images must be of equal size')
        self.data = images
        self.shape = images[0].shape
        #print(self.pixels)
        self.features = np.zeros((self.pixels.shape[0], len(self.data)))
        for i in range(len(self.data)):
            self.features[:, i] = self.data[i][tuple(self.pixels.T)]
        return self.features

    def reverse(self, labels):
        '''
        Reverse the transformation to build a volumetric scan from the labeled feature set
        :param labels: N length array of labels
        :return: 3d array of labeled voxels
        '''
        output = np.zeros(self.shape)
        output[tuple(self.pixels.T)] = labels
        return output
