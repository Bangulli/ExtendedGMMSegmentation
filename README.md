# Extended GMM Segmentation
## Introduction
This repository contains python scripts and classes to perform Gaussian Mixture based clustering on MRI images.
A gaussian mixture is a clustering model that can model elliptical clusters in the feature space, since it is not just using the absolute distance from the cluster center, like a kmeans would.
It is extended with support for a probabilistic atlas and tissue models and was used on Brain images, provided by the UdG in a Medical Image Segmentation and Applications assignment.

## Content

## Classes
A description of the classes in the package
### GMM
The GMM class, supports kmeans, atlas, tissue model and combined initialization.
Optimizes the clusters using the Expectation Maximization algorithm
Support posterior assignment using prior info from Atlas or Tissue Model
Supports informed EM by applying prior information on the weights after the E-Step.
Example usage in the extended_gmm_usage.py script. <br>

#### Parameters:
- k: int, number of clusters
- atlas = Atlas object from the models package, optional
- max_iter int, number of iterations to stop at if convergence is not reached, default = 100
- verbose, bool
    