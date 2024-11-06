import numpy as np
import math
import nibabel as nib
import os
from PrettyPrint import figures
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter
import warnings

class Scorer():
    '''
    Object to assign the random order of cluster labels to the labels of the ground truth by dice score and produce resulting score
    '''
    def __init__(self, gt, predictions, image):
        '''
        Constructor.
        :param gt: nd array, ground truth
        :param predictions: 1d array of labels for cluster presentation (scatter plot) just the output of the gmm
        :param image: 3d array, the volumetric image of labels
        '''
        self.gt = gt
        self.predictions = predictions
        self.image = image

    def relabel(self):
        '''
        Find best label assignment
        :return: relabled image 3d array, predictions 1d array and the winning score list of scores for each tissue
        '''
        score, combi = self._find_best()
        img, pred = self._reassign(combi)
        return img, pred, score

    def relabel_for_bayes(self):
        score, combi = self._find_best()
        return combi

    def _find_best(self):
        '''
        Relabel in each possible configuration and compure score, save best config and score
        :return: best score (list), best config (dict)
        '''
        dicts = [
            {1: 1, 2: 2, 3: 3},
            {1: 1, 2: 3, 3: 2},
            {1: 2, 2: 3, 3: 1},
            {1: 2, 2: 1, 3: 3},
            {1: 3, 2: 1, 3: 2},
            {1: 3, 2: 2, 3: 1}
        ]
        mean = 0
        best_dict = {1:1, 2:2, 3:3}
        best_score = 0
        for elem in dicts:
            d = self._dicted_dice(elem)
            m = np.mean(d)
            if m > mean:
                mean=m
                best_dict=elem
                best_score=d

        print('Best combination has average dice score of:', mean)
        return best_score, best_dict

    def _reassign(self, combi):
        '''
        Reassign the labels according to the best configuration
        :param combi: dictionary of configuration
        :return: relabeled image 3d array, relabeled predictions 1d array
        '''
        img = np.zeros(self.image.shape)
        pred = np.zeros(self.predictions.shape)
        for elem in combi.keys():
            img[self.image == elem] = combi[elem]
            pred[self.predictions == elem] = combi[elem]
        return img, pred

    def _dicted_dice(self, combi):
        '''
        compute the dice score for a given configuration
        :param combi: dictionary of configuration
        :return: list of score for each tissue
        '''
        worker = np.zeros(self.image.shape)
        for elem in combi.keys():
            worker[self.image == elem] = combi[elem]
        labels = np.unique(worker)
        Dices = []
        for elem in labels:
            if elem == 0:
                continue
            i = np.sum(np.bitwise_and(worker == elem, self.gt == elem))
            u = np.sum(worker == elem) + np.sum(self.gt == elem)
            dice = (i * 2) / u
            Dices.append(dice)
        return Dices