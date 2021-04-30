import os
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class neural_network():
    """
    Multi Layered Perceptron for binary classification. 
    
    Parameters -
        Activation Functions - Sigmoid, ReLU 
        Updating Rule - Stochastic Gradient Descent
        Cost Function - Cross Entropy
    """
    
    def __init__(self, nn_architecture):
        """
        Takes in dictionary of input, hidden and output dimensions.
        """
        self.nn_architecture = nn_architecture
    
    def sigmoid(self, z):
        """
        Sigmoid Activation Function
             - 
        """
        return 1/(1+np.exp(-z))

    def relu(self, z):
        """
        ReLU Activation Function
            - 
        """
        return np.maximum(0,z)

    def sigmoid_backward(self, da, z):
        """
        Sigmoid derivative Function
            - 
        """
        sig = self.sigmoid(z)
        return da * sig * (1 - sig)

    def relu_backward(self, da, z):
        """
        ReLU derivative function
            - 
        """
        dz = np.array(da, copy = True)
        dz[z <= 0] = 0
        return dz

    
    def get_cost_value(self, Y_hat, Y):
        """
        Cost Function
            - Cross Entropy Loss: Fit for Binary Classification
            Given as:
                    cost = -1/m*(y*log(y_hat) + (1-y)*log(1-y_hat))
            - Outputs probabalities
        """
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    
    def convert_prob_into_class(self, probs):
        """
        Convert Probabilites into Binary Classes, '0' or '1', with threshold
        of 0.5 probability value.
        """
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_
    
    def get_accuracy_value(self, y_hat, y):
        """
        Calculate accuracy by averaging all values matching in predicted(Y_hat) 
        and actual(Y).
        """
        y_hat_ = self.convert_prob_into_class(y_hat)
        return (y_hat_ == y).all(axis=0).mean()

    '''LAYERS'''
    def init_layers(self, nn_architecture, seed = 101):
        """
        Initializing Layers with input, hidden and output dimensions.
            - Input : Architecture dictionary of input dimension, 
                      output dimensions and hidden dimensions.
            - Output : Parameters values in a list
        """
        np.random.seed(seed)
        params = {}

        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            layer_in_size = layer["input_dim"]
            layer_out_size = layer["output_dim"]

            params['w' + str(layer_idx)] = np.random.randn(layer_out_size, layer_in_size) * 0.1
            params['b' + str(layer_idx)] = np.random.randn(layer_out_size, 1) * 0.1
        
        return params

    
    def single_layer_forward_prop(self, a_prev, w_curr, b_curr, activation = "relu"):
        """
        Forward propagation carried out in single layer
            - Input :
            - Output : 
        """
        z_curr = np.dot(w_curr, a_prev) + b_curr 
        
        if activation is "relu":
            activation_function = self.relu
        elif activation is "sigmoid":
            activation_function = self.sigmoid
        else:
            raise Exception('Unsupported activation function') 
        return activation_function(z_curr), z_curr

    def full_forward_propagation(self, X, params, nn_architecture):
        """
        Applies previously defined forward propagation for single layer to 
        complete architecture's layers with the chosen activation funciton.
            - Input: 
            - Output: 
        """
        memory = {}
        a_curr = X
        
        for idx, layer in enumerate(nn_architecture):
            layer_idx =  idx + 1
            a_prev = a_curr 
            activation_ = layer["activation"] #current activation function in the layer
            w_curr = params["w" + str(layer_idx)] 
            b_curr = params["b" + str(layer_idx)]
            a_curr, z_curr = self.single_layer_forward_prop(a_prev, w_curr, b_curr, activation_)
            
            memory["a" + str(idx)] = a_prev 
            memory["z" + str(layer_idx)] = z_curr
            
        return a_curr, memory 
        
    def single_layer_back_prop(self, da_curr, w_curr, b_curr, z_curr, a_prev, activation = "relu"):
        """
        Backpropagation step in single layer
            -Input:
            -Output:
        """
        m = a_prev.shape[1]

        if activation == "relu":
            backward_activation_func = self.relu_backward
        elif activation == "sigmoid":
            backward_activation_func = self.sigmoid_backward
        else:
            raise Exception('Unsupported activation function')        
        
        dz_curr = backward_activation_func(da_curr, z_curr)
        dw_curr = np.dot(dz_curr, a_prev.T) / m
        db_curr = np.sum(dz_curr, axis=1, keepdims=True) / m
        da_prev = np.dot(w_curr.T, dz_curr)

        return da_prev, dw_curr, db_curr

    def full_backward_propagation(self, y_hat, y, memory, params, nn_architecture):
        """
        Backpropagation
        """
        grads = {}
        m = y.shape[1]
        y = y.reshape(y_hat.shape)
        
        da_prev = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat)); #SGD Initial parameter
        
        for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
            
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
            da_curr = da_prev

            a_prev = memory["a" + str(layer_idx_prev)]
            z_curr = memory["z" + str(layer_idx_curr)]

            w_curr = params["w" + str(layer_idx_curr)]
            b_curr = params["b" + str(layer_idx_curr)]

            da_prev, dw_curr, db_curr = self.single_layer_back_prop(da_curr, w_curr, b_curr, z_curr, a_prev, activ_function_curr)

            grads["dw" + str(layer_idx_curr)] = dw_curr
            grads["db" + str(layer_idx_curr)] = db_curr

        return grads

    def update(self, params, grads, nn_architecture, learning_rate):
        """
        Update Parameters
        """
        for layer_idx, layer in enumerate(nn_architecture, 1):
            params["w" + str(layer_idx)] -= learning_rate * grads["dw" + str(layer_idx)]        
            params["b" + str(layer_idx)] -= learning_rate * grads["db" + str(layer_idx)]
        return params

    def train(self, X, y, nn_architecture, epochs, learning_rate, flag=False, callback=None):
        """
        Training Function  
        """
        params = self.init_layers(nn_architecture, 2)
        cost_history = []
        accuracy_history = []

        for i in range(epochs):
            y_hat, cache = self.full_forward_propagation(X, params, nn_architecture)

            cost = self.get_cost_value(y_hat, y)
            cost_history.append(cost)
            
            accuracy = self.get_accuracy_value(y_hat, y)*100
            accuracy_history.append(accuracy)

            grads = self.full_backward_propagation(y_hat, y, cache, params, nn_architecture)
            params = self.update(params, grads, nn_architecture, learning_rate)

            if(i % 50 == 0):
                if(flag):
                    print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
                if(callback is not None):
                    callback(i, params)

        return params, cost_history, accuracy_history
    

if __name__ == "__main__":
    nn_architecture = [
        {"input_dim": 2, "output_dim": 100, "activation": "relu"},
        {"input_dim": 100, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 1, "activation": "sigmoid"},
    ]

    model = neural_network(nn_architecture)
    
    X, y = make_moons(n_samples = 1000 , noise=0.2, random_state=100) #Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=42) #Splitting the data    
    parameters, cost_history, accuracy_history= model.train(np.transpose(X_train), 
                                                            np.transpose(y_train.reshape((y_train.shape[0], 1))), 
                                                            nn_architecture, 10000, True, 0.01)
