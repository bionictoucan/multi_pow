# Estimating the Flow-function Coefficient of Pharmaceutical Materials from Morphologi G3 Images Using Deep Learning
## John A. Armstrong, 24/01/2023
[![Code style:
black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![DOI](https://zenodo.org/badge/577694581.svg)](https://zenodo.org/badge/latestdoi/577694581)

The following repository contains the code for the building, training and
testing of a simple convolutional neural network architecture (VGG-11 with batch
norm) to learn how to classify images of pharmaceutical materials from Malvern's
[Morphologi G3](https://www.malvernpanalytical.com/en/support/product-support/morphologi-range/morphologi-g3) instrument into three classes of flow-function coefficient (FFc)
following Jernike's classification scheme:

1. Cohesive, FFc &leq; 4
2. Easy flowing, 4 < FFc < 10
3. Free flowing, FFc &geq; 10

The FFc measurements used for training and testing were taken using Freeman
Technology's FT4 Powder Rheometer with its [shear cell
accessory](https://www.freemantech.co.uk/powder-testing/ft4-powder-rheometer-powder-flow-tester/shear-testing).
Please note that all deep learning work is implemented in [PyTorch](https://pytorch.org/).

The important code this work is separated across 6 files:

* `data_prep.py` contains functions for preparing a dataset for either training
  or testing with the user only needing to provide paths to the images and to
  the class labels
* `dataset.py` contains a custom PyTorch dataset for dealing with the images.
* `model.py` contains a simple function which uses PyTorch's
  [torchvision](https://pytorch.org/vision/stable/index.html) implementation of
  the VGG-11 network with batch normalisation and replaces the first
  convolutional layer to one with 1 input channel (since our images are
  greyscale) and replaces the classifier section with the same number of
  fully-connected layers but mapping to the number of classes we need from the
  model. The model is initialised with torchvision's pretrained ImageNet1K
  weights with the custom layers then initialised by Kaiming initialisation in
  the weights and zeros in the biases.
* `testing.py` contains a variety of functions for external testing of the model
  as well as combining two binary models into a multiclass model and performing
  the majority vote on image segments to get an overall prediction of the whole
  image.
* `training.py` contains objects for training and validating the networks
  depending on whether it is a binary problem or a multiclass problem.
* `utils.py` contains the method for segmenting the images into 1024&times;1024
  segments to be treated as independent samples by the model. This is needed as
  the original images from the G3 used are of the entire microscope slide and
  even after downsampling are still huge (~6k&times;4k pixels).

The `examples` folder contains two Jupyter notebooks which demonstrate how to
train a network and how to test a network.