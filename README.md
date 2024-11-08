# Extended GMM Segmentation
## Content
- [Introduction](##Introduction)
- [Classes](##Classes)
- [Examples](##Examples)
- [License](##License)
## Introduction
This repository contains python scripts and classes to perform Gaussian Mixture based clustering on MRI images.
A gaussian mixture is a clustering model that can model elliptical clusters in the feature space, since it is not just using the absolute distance from the cluster center, like a kmeans would.
It is extended with support for a probabilistic atlas and tissue models and was used on Brain images, provided by the UdG in a Medical Image Segmentation and Applications assignment.

## Classes
A description of the classes in the package
### GMM
The GMM class, supports kmeans, atlas, tissue model and combined initialization.
Optimizes the clusters using the Expectation Maximization algorithm
Support posterior assignment using prior info from Atlas or Tissue Model
Supports informed EM by applying prior information on the weights after the E-Step.
Example usage in the [example script](extended_gmm_usage.py). <br>

#### Parameters:
- k: int, number of clusters
- atlas = Atlas object from the models package, optional
- max_iter int, number of iterations to stop at if convergence is not reached, default = 100
- verbose, bool, controls if detailed parameter changes are printed to the console, default False

### Scorer

### FeatureTransformer

### Atlas

### TissueModel

## Examples

### Atlas building

### Tissue Model building

### Extended GMM usage

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

    