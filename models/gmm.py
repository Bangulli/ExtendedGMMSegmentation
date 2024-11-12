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
import warnings

class GMM:
    '''
    Implementation of the Gaussian Mixture Model according to the presentation provided by the MAIA Medical Image Segmentation and Applications course.
    Supports atlas and tissue models for prior information
    '''
    def __init__(self, k, max_iter=100, init='kmeans', prior=None, verbose=False):
        '''
        Constructor. Initializes the GMM with parameters
        :param k: int, number of clusters
        :param atlas: instance of an atlas model for prior knowledge
        :param max_iter: int, maximum number of optimization iterations
        :param init: String, initialization method
        :param verbose: bool detailed reporting of parameters during em
        :param tissue_model: instance of a tissue model for prior knowledge
        '''
        self.k = k
        self.max_iter = max_iter
        self.init = init
        self.prior = prior
        self.verbose = verbose
        return

    def fit(self, image, mask, influence_frq=0):
        '''
        Fit function. Optimizes the parameters of the GMM to the given dataset
        :param X: 2d array, The dataset
        :return: None
        '''
        self.ft = FeatureTransformer()
        self._init(image, mask)
        if influence_frq == 0:
            self._em()
        else:
            self._influence_em(influence_frq)

    def fit_transform(self, image, mask, posteriori=False, influence_frq=0):
        '''
        Fit and return the labeled data
        :param X: 2d array, the dataset to be fit and predicted
        :return: 1d array, predictions
        '''
        self.fit(image, mask, influence_frq)
        if posteriori:
            return self.ft.reverse(self._posterior_assign()+1)
        else:
            return self.ft.reverse(self._assign(self.weights)+1)

    def _em(self):
        '''
        Optimization iteration method. Runs the Expectation Maximization algorithm to optimize the parameters for the gaussian mixture
        :return: None
        '''
        prog = figures.RunningIndicator()
        for i in range(self.max_iter):
            prog()
            if self.verbose:
                print('Alphas:\n', self.alphas, '\n', 'Means:\n', self.means, '\n', 'Variances:\n', self.variances)
            self._estep()
            self._mstep()
            if self._convergence():
                break
        print('\n')
    def _influence_em(self, influence_frq):
        '''
        Optimization iteration method. Runs the Expectation Maximization algorithm to optimize the parameters for the gaussian mixture
        :param influence_frq: the frequency of how often the prior knowledge is applied to the responsibilities over the iterations of the em algorithm
        :return: None
        '''
        prog = figures.RunningIndicator()
        for i in range(self.max_iter):
            prog()
            if self.verbose:
                print('Alphas:\n', self.alphas, '\n', 'Means:\n', self.means, '\n', 'Variances:\n', self.variances)
            self._estep()
            if i%influence_frq == 0:
                self.weights = self.weights * self.prior_weights
            self._mstep()
            if self._convergence():
                break
        print('\n')

    def _estep(self): # the brainfck step
        '''
        Computes the E-Step of the EM algorithm.
        1st: Compute the Probability Density Function (PDF) for every sample and every cluster
        2nd: Compute the responsibility of every point to every cluster. Normalized so the sum of the probabilities is 1
        :return: None
        '''
        self.weights = np.zeros((self.X.shape[0], self.k))
        for k in range(self.k):
            ##################################### step 1: density
            cov = self.variances[k] # get cov matrix
            #cov += np.eye(self.dims) * 1e-6 # regularization to avoid numerical errors
            #print(cov)
            cov_inv = np.linalg.inv(cov) # prepare inverse for pdf formula
            det = np.linalg.det(cov) # prepare determinant for pdf formula
            diff = self.X - self.means[k] # get difference for each point to the current cluster mean
            # formula 2 slide 56
            # build components
            # i missed a bracket in the constant that fucked it all up :(((((((((((((((((((((
            constant = 1/(((2*np.pi)**(self.dims/2))*(det**(1/2)))
            ''' # my own implementation but its slow af so i asked chatgpt to optimize            
            for i in range(self.samples):
                diff = self.X[i, :] - self.means[k]
                exp = -0.5 * (diff.T @ cov_inv @ diff)
                self.p[i, k] = constant * np.exp(exp)
            '''
            # optimized implementation of the above by chatgpt
            exp = -0.5 * np.einsum('ij,jk,ik->i', diff, cov_inv, diff) # np.einsum vectorizes the operation and is more efficient than pointwise computation.
            self.p[:, k] = constant * np.exp(exp) # computes the gaussian density for each point

            ##################################### step 2: membership
            self.weights[:, k] = self.p[:, k] * self.alphas[k] # part 1 of formula 1 slide 56

        # normalize weights
        self.weights /= self.weights.sum(axis=1, keepdims=True) # part 2 of formula 1 slide 56

    def _mstep(self): # the easy step
        '''
        Computes the M-Step of the EM algorithm.
        1st: Assign each sample to a cluster according to the highest responsibility
        2nd: Update mixing coefficients = alpha
        3rd: Update means
        4th: Update covariance matrices of the Gaussian distributions
        :return: None
        '''
        for k in range(self.k):
            N_k = np.sum(self.weights[:, k])
            # update alphas
            self.alphas[k] = N_k/self.samples # formula slide 57
            # update means
            self.means[k, :] = np.sum((self.weights[:, k].reshape(-1, 1)*self.X), axis=0)/N_k # formula 1 slide 58
            # update covariance matrices
            ''' #my own implementation but its slow af so i asked chatgpt to optimize
            _sum = np.zeros(self.variances[k].shape)
            for j in range(self.samples): # formula 2 slide 58 works pointwise, this loop is the summation function. its pretty slow though, so to be optimized
                diff = self.X[j, :] - self.means[k, :]
                _sum += self.weights[j, k]*np.outer(diff, diff)
            self.variances[k] = _sum/N_k# formula 2 slide 58
            '''
            # optimization of the above implementation by chatgpt
            diff = self.X - self.means[k]  # Difference between all data points and the mean
            weighted_outer_products = (self.weights[:, k, np.newaxis] * diff).T @ diff
            self.variances[k] = weighted_outer_products / N_k # Update the covariance matrix


    def _assign(self, X): # put in seperate method in case the same gmm is used to predict on unkown data
        '''
        Assign the labels to the data according to the largest responsibility
        :param X: Nxk array of responsibilities
        :return: Nx1 array of assigned labels
        '''
        res = np.argmax(X, axis=1)
        return res

    def _posterior_assign(self):
        '''
        Assign the labels to the data according to the largest responsibility
        uses the prior information of the tissue model or atlas to refine the output
        :return:
        '''
        res = np.argmax((self.prior_weights * self.weights), axis=-1)
        return res

    def _make_cov(self, mask): # makes covariance matrix for n-feature dimensions
        """
        Build the covariance matrix of a cluster in the dataset
        :param mask: a binary array of samples that belong to a cluster
        :return: dimsxdims covariance matrix
        """
        cov = np.zeros((self.dims, self.dims))
        data = self.X[mask, :]
        var = np.var(data, axis=0)
        for i in range(self.dims):
            cov[i, i] = var[i]
        return cov

    def _init(self, image, mask): # initializes all parameters according to the passed method. For now just supports kmeans
        '''
        Initialize the parameters of the GMM
        kmeans:
            alphas = 1/k
            means & covariance matrix according to passed init method
        other:
            uses probability information of the tissue or atlas model as initial cluster responsibilities and
            performs an M-step with that to initialize the parameters
        :return: None
        '''
        # get data
        if isinstance(image, list):
            self.X = self.ft.transform(image, mask)
        else:
            self.X = self.ft.transform([image], mask)
        # prepare prior weights, regardless of usage or not
        if self.prior is not None:
            # get probabilistic
            seg = self.prior.soft_segment(image)
            anatomical = self.prior.segment(image)
            self.prior_weights = self.ft.retransform([seg[..., 0], seg[..., 1], seg[..., 2]])
        # init the hard one
        if self.init == 'kmeans':
            self.dims = self.X.shape[1]
            self.samples = self.X.shape[0]
            # init easy
            self.alphas = [1 / self.k] * self.k  # mixing coefficients
            self.p = np.zeros((self.samples, self.k))
            self.loglikelihood = 0
            # init clusters
            model = KMeans(n_clusters=self.k)
            model.fit(self.X)
            assigned = model.labels_
            # init distributions
            self.means = model.cluster_centers_
            self.variances = []
            for i in range(self.k):
                cov = self._make_cov(assigned==i)
                self.variances.append(cov)

            warnings.warn('KMeans initialization without prior knowledge requires a Scorer object to use at the output, to reassign labels to ground-truth')



        # init with priors
        elif self.init == 'prior':
            self.dims = self.X.shape[1]
            self.samples = self.X.shape[0]
            # init easy
            self.p = np.zeros((self.samples, self.k))
            self.loglikelihood = 0
            self.alphas = list(np.zeros(self.k))
            self.variances = [np.zeros((self.dims, self.dims))]*self.k
            self.means = np.zeros((self.k, self.dims))
            # assign weights and mstep.
            self.weights = self.prior_weights.copy()
            self._mstep()

        else:
            raise NotImplementedError('Unknown init method')



    def _convergence(self, tol=1e-6):
        '''
        Check for convergence according to a tolerance range. Computes the loglikelihood of the model and compares it to the
        previous iteration. If the difference is less than the tolerance return true
        :param tol: the tolerance of the criteria
        :return: bool, True if convergence is reached, False if not.
        '''
        inner_sums = np.dot(self.p, self.alphas)
        inner_sums = np.maximum(inner_sums, 1e-300)
        ll = np.sum(np.log(inner_sums))
        if abs(self.loglikelihood - ll) < tol:
            print('\nConvergence found')
            return True
        else:
            #print(ll)
            self.loglikelihood = ll
            return False

