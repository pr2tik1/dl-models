import os
import struct 
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

from data import load_data, visualize_data

#Loading the data
train_x,  train_y, test_x, test_y = load_data()


def enc_one_hot(y, num_labels = 10):
	'''
	Function to One Hot Encode the labels 
	Arguments:
		- Input : Label and number of labels
		- Output: Matrix of one hot encoded values
	'''
	one_hot = np.zeros((num_labels, y.shape[0]))
	for i, val in enumerate(y):
		one_hot[val, i] = 1.0
	return one_hot


def sigmoid(z):
	'''
	Activation Function
	'''
	return expit(z)


def sigmoid_gradient(z):
	'''
	Activation Function's Gradient 
	'''
	result = sigmoid(z)
	return result*(1-result)


def visualize_sigmoid():
    '''
	Visualizing acitivation function
    '''
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

def calc_cost(y_enc, output):
	"""
    Cost Function
    """
	cost = np.sum(-y_enc*np.log(1-output) - (1-y_enc)*np.log(1-output))
	return cost

def add_bias_unit(X, where):
    """
    Adding Bias in row and columns
    """
    if where == 'column':
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
    elif where == 'row':
        X_new = np.ones((X.shape[0] + 1, X.shape[1]))
        X_new[1:, :] = X
    return X_new

def init_weights(n_features, n_hidden, n_output):
    """
    Initializing weights with normal distribution
    """
    w1 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_features+1))
    w1 = w1.reshape(n_hidden, n_features+1)
    w2 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_hidden+1))
    w2 = w2.reshape(n_hidden, n_hidden+1)
    w3 = np.random.uniform(-1.0, 1.0, size=n_output*(n_hidden+1))
    w3 = w3.reshape(n_output, n_hidden+1)
    return w1, w2, w3

def feed_forward(x, w1, w2, w3):
    """
    Forward propagation
    """
    a1 = add_bias_unit(x, where='column')#Adding Bias units
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)
    
    a2 = add_bias_unit(a2, where='row')
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)
    
    a3 = add_bias_unit(a3, where='row')
    z4 = w3.dot(a3)
    a4 = sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4

def predict(x, w1, w2, w3):
    """
    Prediction Function
    """
    a1, z2, a2, z3, a3, z4, a4 = feed_forward(x, w1, w2, w3)
    y_pred = np.argmax(a4, axis=0)
    return y_pred

def calc_grad(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    """
    Gradient Calculation
    """
    delta4 = a4 - y_enc
    z3 = add_bias_unit(z3, where='row')
    
    delta3 = w3.T.dot(delta4)*sigmoid_gradient(z3)
    delta3 = delta3[1:, :]
    
    z2 = add_bias_unit(z2, where='row')
    
    delta2 = w2.T.dot(delta3)*sigmoid_gradient(z2)
    delta2 = delta2[1:,:]

    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)

    return grad1, grad2, grad3
    

if __name__ == '__main__':
	print('What number would like to visualize?(0-9)')
	number = int(input())
	visualize_data(number, train_x, train_y)
	
	print('Would like to view sigmoid? (y/n)')
	response = str(input())
	if response=='y':
		visualize_sigmoid()
	else :
		print('Oh!, Ok Thanks!')