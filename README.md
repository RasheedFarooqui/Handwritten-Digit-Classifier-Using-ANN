# Handwritten-Digit-Classifier-Using-ANN

# MNIST Digit Classification

This project implements a neural network model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model is trained to predict the digit represented in a given image based on the pixel values.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Results](#results)

## Overview

The MNIST dataset contains 70,000 images of handwritten digits (0-9), split into a training set of 60,000 images and a test set of 10,000 images. The model uses a fully connected neural network to classify these images, achieving high accuracy.

## Dataset

The MNIST dataset is available through the TensorFlow Keras library and consists of:

- **Training set**: 60,000 images of handwritten digits
- **Test set**: 10,000 images for evaluating the model's performance

## Installation

To run this project, you will need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`

## Results

After training the model for 10 epochs with a validation split of 20%, the model achieved an accuracy of **97%** on the test dataset. The performance of the model is visualized using a confusion matrix, which helps identify the number of correct and incorrect predictions for each digit.



