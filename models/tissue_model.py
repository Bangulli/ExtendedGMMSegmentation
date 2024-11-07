import numpy as np
import math
import nibabel as nib
import os
import matplotlib.pyplot as plt
from PrettyPrint import figures
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter
import warnings
class TissueModel:
    '''
    Class to generate and use a tissue model based on intensity of volumetric scans
    '''
    def __init__(self, labels=3, norm='linear'):
        '''
        Constructor
        :param labels: number of tissues in the model, default 3 for cerebrospinal fluid, grey matter and white matter
        :param norm: image intensity normalization strategy, default 'linear'
        '''
        self.labels = labels
        self.norm = norm

    def fit(self, images, gts):
        '''
        Fit the tissue model accoring to the passed images and ground truths
        :param images: list of volumetric images (=3D numpy array)
        :param gts: list of ground truth labels (=3D numpy array)
        :return:
        '''
        if len(images) != len(gts):
            raise ValueError('Images and Gts must have the same amount')
        self.images = images
        self.gts = gts

        self._norm_imgs()
        self._build()

    def _build(self):
        '''
        Build the tissue model by counting and averaging intensities for each label
        :return:
        '''
        self.L_prob = np.zeros((self.labels, 4096), dtype=np.float64)

        for i in range(len(self.images)):  # Iterate over images
            for l in range(self.labels):  # Iterate over label types
                # Apply mask to select pixels of a specific label in the current image
                w = self.images[i][self.gts[i] == (l+1)]  # Mask for current label

                # Compute histogram for the selected tissue type
                hist, _ = np.histogram(w, bins=4096, range=(0, 4095), density=False)

                # Accumulate the histogram for the current label across images
                self.L_prob[l, :] += hist

        #self.L_prob[self.L_prob<=len(self.images)]=0#filter out noise
        # Normalize by summing along the columns to avoid zero division
        axis_sum = np.sum(self.L_prob, axis=0).astype(np.float64)
        axis_sum[axis_sum == 0] = 1  # Replace zeros with ones to avoid division by zero

        # Normalize each label's histogram by total sum per bin
        for l in range(self.labels):
            self.L_prob[l, :] /= axis_sum
            self.L_prob[l, :] = median_filter(self.L_prob[l, :], size=3)


    def _comp_ref_hist(self):
        '''
        Used in hist normalization strategy, computes the reference histogram based on the underlying data
        :return:
        '''
        ref = np.zeros(4096, dtype=np.float64)
        for img in self.images:
            hist, _ = np.histogram(img, bins=4096, range=(0, 4095), density=True)
            ref += hist
        self.ref_hist = ref/len(self.images)

    def _norm_imgs(self):
        '''
        Helper method to normalize an entire image set
        :return:
        '''
        if self.norm == 'hist':
            self._comp_ref_hist()
            for i in range(len(self.images)):
                print('Histmatching image:', i)
                self.images[i] = self._hist_match(self.images[i])
        elif self.norm == 'linear':
            for i in range(len(self.images)):
                low = np.min(self.images[i])
                high = np.max(self.images[i])
                min_range = 0
                max_range = 4059
                self.images[i] = (self.images[i] - low) / (high - low) * (max_range - min_range) + min_range
                self.images[i] = self.images[i].astype(np.uint16)
        elif self.norm == 'none':
            pass
        else:
            raise ValueError('Normalization method not recognized')

    def _norm_img(self, img):
        '''
        helper method to normalize a single image for segmentation
        :param img:
        :return:
        '''
        if self.norm == 'hist':
            self._comp_ref_hist()
            img = self._hist_match(img)
        elif self.norm == 'linear':
            low = np.min(img)
            high = np.max(img)
            min_range = 0
            max_range = 4059
            img = (img - low) / (high - low) * (max_range - min_range) + min_range
            img = img.astype(np.uint16)
        elif self.norm == 'none':
            pass
        else:
            raise ValueError('Normalization method not recognized')
        return img

    def _hist_match(self, img, L=4096):  # histogram matchin of image to target average histogram
        '''
        matches the histogram of the current image to a precomputed target histogram
        the target is the average cdf of all images in the training set.
        this is the code that prof bria wrote in class, translated by chatgpt and then modified to take
        a histogram instead of an image
        anyway, sometimes gives a warning when information is lost because float 32 doesnt provide enough decimals but we ignore that, cause its just minute information that shouldnt change much
        '''
        # Calculate source CDF
        hist_s, _ = np.histogram(img, bins=L, range=(0, L-1), density=True)
        cdf_s = hist_s.cumsum()
        cdf_s = np.ma.masked_equal(cdf_s, 0)
        cdf_s = (cdf_s - cdf_s.min()) * (L-1) / (cdf_s.max() - cdf_s.min())
        cdf_s = np.ma.filled(cdf_s, 0).astype('uint16')

        # Calculate target CDF
        cdf_t = self.ref_hist.cumsum()
        cdf_t = np.ma.masked_equal(cdf_t, 0)
        cdf_t = (cdf_t - cdf_t.min()) * (L-1) / (cdf_t.max() - cdf_t.min())
        cdf_t = np.ma.filled(cdf_t, 0).astype('uint16')

        # Calculate transform
        LUT = np.zeros(L, dtype='uint16')
        for i in range(L):
            diff = np.abs(cdf_s[i] - cdf_t[0])
            for j in range(1, L):
                new_diff = np.abs(cdf_s[i] - cdf_t[j])
                if new_diff < diff:
                    diff = new_diff
                    LUT[i] = j

        # Apply transform
        img_normalized = np.clip(img, 0, L - 1).astype('uint16')
        result = LUT[img_normalized]

        return result

    def save(self, path):
        '''
        Saves the tissue models attributes in a target location, creates the folder if necessary
        :param path: String, path to the target location, optionally with new fodler name
        :return:
        '''
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'tissue_model.npy'), self.L_prob)
        if self.norm == 'hist':
            np.save(os.path.join(path, 'ref_hist.npy'), self.ref_hist)

    def load(self, path):
        '''
        Load the tissue model from a location
        :param path: path to the model
        :return:
        '''
        self.L_prob = np.load(os.path.join(path, 'tissue_model.npy'))
        if self.norm == 'hist':
            self.ref_hist = np.load(os.path.join(path, 'ref_hist.npy'))

    def segment(self, image):
        '''
        Segments an image based on the tissue model by assigning labels to intensities
        :param image: 3D numpy array, the image to segment
        :return: 3D numpy array, the segmented image
        '''
        image = self._norm_img(image)
        segmentation = self.L_prob[:, image].transpose(1, 2, 3, 0)
        mask = np.sum(segmentation, axis=-1)!=0
        segmentation = np.argmax(segmentation, axis=-1)
        return segmentation[mask]+1

    def soft_segment(self, image):
        '''
        Assigns probabilites of a voxel belonging to a tissue
        :param image: 3d numpy array, the image to segment
        :return: 4d numpy array, the segmented image
        '''
        image = self._norm_img(image)
        segmentation = self.L_prob[:, image].transpose(1, 2, 3, 0)
        #print(segmentation.shape)
        return segmentation

    def show(self):
        data = self.L_prob

        # X-axis values from 0 to 4095
        x = np.arange(4096)

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot each line
        for i in range(3):
            plt.plot(x, data[i], label=f'label {i}')

        # Set x and y limits
        plt.xlim(0, 4095)
        plt.ylim(0, 1)

        # Add labels, title, and legend
        plt.xlabel("X-axis (0-4095)")
        plt.ylabel("Y-axis (0-1)")
        plt.title("Tissue model intensity probabilities")
        plt.legend()

        # Display the plot
        plt.show()
