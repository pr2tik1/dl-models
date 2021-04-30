# Exploring Neural Networks

## 1. Neural Network Scratch implementation
- https://github.com/pr2tik1/neural-networks/tree/master/mlp-numpy

## 2. MNIST Multi-Class Image Classification

This part of the repository contains multiclass classification of MNIST data using Classification models for comparitive study.

<details open>

## About Data 
1. MNIST Handwritten Digits dataset : It contains images of digits taken from a variety of scanned documents, normalized in size and centered. Each image is a 28 by 28 pixel square (784 pixels total), values of image pizels ranging from 0-255. The dataset contains 40,000 images for model training and 10,000 images for the evaluation of the model. Source : Kaggle or Framework(PyTorch,sklearn).

2. Fashion-MNIST dataset :  It is a dataset of Zalando's article images — consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. The dataset serves as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

## Handwritten Digits

<details open>
<summary> Digit Recognizer using Numpy (Scratch Implementation)</summary> 

+ Multi Layer Perceptron <a href="https://github.com/pr2tik1/mnist/blob/master/src/train.py">: Training code</a>
+ Data : MNIST dataset (http://yann.lecun.com/exdb/mnist/)
+ Accuracy (Validation) : 95 %

</details>

<details open>
<summary> Digit Recognizer using PyTorch</summary> 

+ Multi Layer Perceptron <a href="https://github.com/pr2tik1/Digit-recognizer/blob/master/mlp-digits-mnist.ipynb">: Code</a>
+ Loss Function : Cross Entropy
+ Stochastic Gradient Descent
+ Data : Kaggle dataset (https://www.kaggle.com/c/digit-recognizer)
+ Accuracy (Validation) : 98.07 %
+ Kaggle Score : 0.98085 

</details>

## Fashion-MNIST

<details open>
<summary>MLP classifier: PyTorch</summary>
      
+ Multi Layer Perceptron<a href="https://github.com/pr2tik1/Digit-recognizer/blob/master/mlp-fashion-mnist.ipynb">: Code</a> 
+ Loss Function : Cross Entropy
+ Stochastic Gradient Descent

</details>

<details open>
<summary>MLP classifier: Tensorflow</summary>
      
+ Multi Layer Perceptron<a href="https://github.com/pr2tik1/Digit-recognizer/blob/master/tf-mnist.ipynb">: Code</a> 
+ Loss Function : Cross Entropy
+ Adam Optimizer

</details>


<details open>
<summary>LeNet classifier</summary>
      
+ LeNet[: Code](https://github.com/pr2tik1/mnist/blob/master/LeNet.ipynb) 
+ Loss Function : Cross Entropy
+ Stochastic Gradient Descent
+ CNN 

</details>

</details>

## 3. Understanding Image Classification with simple example 

- https://pr2tik1.github.io/blog/image%20classification/image%20processing/2021/03/04/Simple-Image-Classification.html



## Dependencies :
   1. Numpy
   2. Pickle
   3. json
   4. Scipy
   5. gzip
   7. Pandas
   8. Matploltlib
   9. PyTorch(torch 1.5+, torchvision 0.4.0+)


## Author:
- Pratik Kumar

## References
- http://neuralnetworksanddeeplearning.com/
- https://github.com/udacity (Udacity Deep Learning ND lectures and projects)