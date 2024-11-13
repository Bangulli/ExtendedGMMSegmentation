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

##### fit_transform(image, mask, influence_frq, posteriori) -> segmented image
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
##### relabel() -> image relabeled, _, _
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
##### transform(images, mask) -> list of data
Transforms the volumetric data into the feature space for the GMM. Uses an area of interest to generate the data, essentially performing the skull stripping
- images: list of 3D numpy arrays, the sources of the data, can be multi modality or 3D probability maps for atlas
- mask: 3D bool numpy array, mask for the area of interest
##### retransform(images) -> list of data
Transforms volumetric data into feature space by using the internally stored area of interest, passed from the transform method
ensures consistency in data order.
- images: list of 3D numpy arrays, the sources of the data, can be multi modality or 3D probability maps for atlas

##### reverse(labels) -> volumetric image
Transforms data from feature space back into 3D space. Takes a list of labels and moves them to the corresponding voxel.
- labels: 1xN numpy array, labeled datapoints in the same order as the transformed output

### Atlas
The atlas object, builds or loads a probabilistic atlas from a set of registered images and ground truths.
Models the probability of a voxel to be a certain tissue by using its spatial location.
#### Parameters
- labels: int, the number of labels in the ground truth, optional, default = 4 for bg, csf, wm, gm
#### Methods
##### build(images, gts, affine)
Build a probabilistic and anatomical atlas from an ordered list of registered images and ground truths as well as a reference image.
- images: list of 3D numpy arrays, the registered images of the atlas training set
- gts: list of 3D numpy arrays, the registered ground truths of the atlas training set
- affine: 2D numpy array, the affine transformation matrix of the fixed image of the atlas registration.
##### save(path)
Makes a directory and stores the atlas as nifty files.
- path: String, the path to the save location
##### load(path)
Loads an atlas from disk to use in code.
- path: String, the path to the save location
##### segment(image) -> labeled image
Returns the anatomical atlas. If the atlas is registered to a target image, this corresponds to the segmentation of the image.
- image: discarded parameter for compatibility with the other prior knowledge objects.
##### soft_segment(image) -> probability image
Retruns the probabilistic atlas for the tissues csf, wm and gm. If the atlas is registered to target image,
this corresponds to the tissue probabilities for the image.
- image: discarded parameter for compatibility with the other prior knowledge objects.

### TissueModel
The tissue model object, builds or loads a tissue model from a set of unregistered images and ground truths.
Models the probability of a voxel to be a certain tissue by using its intensity.
#### Parameters
- labels: int, the number of tissues in the ground truth, optional, default = 3 for csf, gm, wm
- norm: String, the normalisation strategy, optional, default='linear', supports 'linear' for linear histogram equalization, 'hist' for histogram matching, or 'none' for no equalization.

#### Methods
##### fit(images, gts)
fits the tissue model to the data. The images and labels do not need to be registered, **but the order of images and gts must be the same.**
- images: list of 3D numpy arrays, the images used in the dataset
- gts: list of 3D numpy arrays, the labels used in the dataset

##### save(path)
Makes a directory and stores the tissue model as numpy files.
- path: String, the path to the save location
##### load(path)
Loads a tissue model from a save location
- path: String, the path to the save location
##### segment(image) -> labeled image
Segment an image by assigning each voxel a label based on its intensity and the tissue model.
- image: 3D numpy array, the image to segment
##### soft_segment(image) -> probability image
Generate a probability map by assigning each voxel a tissue probability for each tissue based on its intensity.
- image: 3D numpy array, the image to segment
##### show()
Makes a plot of the tissue model, intensity vs probability for each label

### TMACombination
Combined tissue model and atlas object. Uses an atlas and a tissue model to combine spatial and intensity information to 
provide probabilities.
#### Parameters
- atlas: Atlas object, the atlas to be used in this combination, atlas must be registered to the current working image.
- tm: TissueModel object, the tissue model to be used in this combination
#### Methods
###### segment(image) -> labeled image
Segment an image by assigning each voxel a label based on its intensity and spatial position.
- image: 3D numpy array, the image to segment
##### soft_segment(image) -> probability image
Generate a probability map by assigning each voxel a tissue probability for each tissue based on its intensity and spatial position.
- image: 3D numpy array, the image to segment

## Examples
A set of example scripts that show the usage of the provided code
### [Atlas building](build_atlas.py)
Shows how to build an atlas using the provided class. Note that the images need to be registered beforehand.
Usage of the provided parameter files is recommended.
### [Tissue Model building](build_tm.py)
Shows how to build a tissue model using the provided class. Does not require any registration.
### [Extended GMM usage](extended_gmm_usage.py)
Shows how to segment an image using the provided class. Note that the atlas has to be registered to each target image individually beforehand.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

    