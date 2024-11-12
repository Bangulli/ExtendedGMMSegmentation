# Extended GMM Segmentation
Atlas and Tissue Model supported GMM clustering for medical images.
## Content
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Classes](#Classes)
- [Examples](#Examples)
- [License](#License)
## Introduction
This repository contains python scripts and classes to perform Gaussian Mixture based clustering on MRI images.
A gaussian mixture is a clustering model that can model elliptical clusters in the feature space, since it is not just using the absolute distance from the cluster center, like a kmeans would.
It is extended with support for a probabilistic atlas and tissue models and was used on Brain images, provided by the UdG in a Medical Image Segmentation and Applications assignment.
## Requirements
This package uses a few external libraries, make sure to have the following packages installed
pip install:
- numpy
- matplotlib
- nibabel
- prettyprintout
- scipy
## Classes
A description of the classes in the package
### GMM
The GaussianMixtureModel class, supports kmeans, atlas, tissue model and combined initialization.
It is a clustering algorithm that fits n clusters to a data distribution. The clusters are Gaussian distributions, which allows this algorithm to model elliptical data distributions.
Optimizes the clusters using the Expectation Maximization algorithm
Support posterior assignment using prior info from Atlas or Tissue Model
Supports informed EM by applying prior information on the weights after the E-Step.
Example usage in the [example script](extended_gmm_usage.py). <br>

#### Parameters:
- k: int, number of clusters
- prior = Prior knowledge object. Must support a segment(image) and a soft_segment(image) function, optional
- max_iter int, number of iterations to stop at if convergence is not reached, default = 100
- verbose, bool, controls if detailed parameter changes are printed to the console, default False
- init, String, the initialization strategy, supports 'kmeans' or 'prior' init. Note that usage of posterior and influence requires prior init for label consistency. Optional, default = 'kmeans'

#### Methods:
##### fit(image, mask, influence_frq)
Fits the gaussian distributions to the dataset. 
- image: 3D numpy array, the image to be segmented, can be a list of images to combine multiple modalities
- mask: 3D numpy array, binary image to provide an area of interest to perform skullstripping
- influence_frq: int, the frequency of prior knowledge influence on the expectation maximization. Frequency in iterations. Optional, default = 0 -> no influence

##### fit_transform(image, mask, influence_frq, posteriori)
Same as fit but returns the segmented image after it is done.
- posteriori: bool, use posterior influence or not, default = False

### Scorer
Brute force label reassignment. Used in case of kmeans initialization to fit the randomly ordered labels to the gt.
Computes the dice score for every possible label combination and picks the best by finding the largest average dice score across labels

#### Parameters:
- gt: 3D numpy array, the labeled ground truth
- predictions: 1xN array of assigned samples, output of the em, not really used anymore. Optional
- image: 3D numpy array, the labeled output of the segmentation algorithm

#### Methods
##### relabel()
performs the scoring and returns the best result
return:
- img: 3D numpy array, the relabeled segmentation image
- predictions: 1xN array, the relabeled predictions, can be discarded
- score: list of float, the dice scores for each label in the best combination

### FeatureTransformer
Transforms the data from image into feature space. Used internally in the GMM class.
#### Parameters:
None
#### Methods:
##### transform(images, mask)
##### retransform(images)
##### reverse(labels)

### Atlas
The atlas object, builds or loads a probabilistic atlas from a set of registered images and ground truths.
Models the probability of a voxel to be a certain tissue by using its spatial location.
#### Parameters
#### Methods
##### build()
##### save()
##### load()
##### segment()
##### soft_segment()

### TissueModel
The tissue model object, builds or loads a tissue model from a set of unregistered images and ground truths.
Models the probability of a voxel to be a certain tissue by using its intensity.
#### Parameters
#### Methods
##### build()
##### save()
##### load()
##### segment()
##### soft_segment()

### TMACombination
Combined tissue model and atlas object. Uses an atlas and a tissue model to combine spatial and intensity information to 
provide probabilities.
#### Parameters
#### Methods
##### build()
##### save()
##### load()
##### segment()
##### soft_segment()

## Examples

### Atlas building

### Tissue Model building

### Extended GMM usage

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

    