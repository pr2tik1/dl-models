import os
import numpy as np
import matplotlib.pyplot as plt
from data import load_data
from model import *
from tqdm import tqdm

def train(X, y, X_t, y_t, epochs, batch, alpha, eta, dec):
    """
    Training Function 
    Arguments: 
        - X      : Training Input
        - y      : Training Labels 
        - X_t    : Test Inputs
        - y_t    : Test Labels
        - epochs : Number of Epochs
        - batch  : Batch size 
        - alpha  : Learning Rate
        - dec    : Weight Decay
    """
    X_copy, y_copy = X.copy(), y.copy()
    y_enc = enc_one_hot(y)
    
    w1, w2, w3 = init_weights(784, 75, 10)

    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)
    total_cost = []
    pred_acc = np.zeros(epochs)

    for i in tqdm(range(epochs)):

        shuffle = np.random.permutation(y_copy.shape[0])
        X_copy, y_enc = X_copy[shuffle], y_enc[:, shuffle]
        eta /= (1 + dec*i)

        mini = np.array_split(range(y_copy.shape[0]), batch)

        for step in mini:
            # feed forward 
            a1, z2, a2, z3, a3, z4, a4 = feed_forward(X_copy[step], w1, w2, w3)
            cost = calc_cost(y_enc[:,step], a4)

            total_cost.append(cost)
            # back prop
            grad1, grad2, grad3 = calc_grad(a1, a2, a3, a4, z2, z3, z4, y_enc[:,step],
                                            w1, w2, w3)
            delta_w1, delta_w2, delta_w3 = eta * grad1, eta * grad2, eta * grad3

            w1 -= delta_w1 + alpha * delta_w1_prev
            w2 -= delta_w2 + alpha * delta_w2_prev
            w3 -= delta_w3 + alpha * delta_w3_prev

            delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3_prev

        y_pred = predict(X_t, w1, w2, w3)
        pred_acc[i] = 100*np.sum(y_t == y_pred, axis=0) / X_t.shape[0]
        #print('Epoch :', i)
    return total_cost, pred_acc, y_pred



if __name__=="__main__":
    
    print("Enter number of epochs(1000):")
    epochs = int(input()) #1000

    print("Enter value for batch size(50):")
    batch = int(input()) #50
    
    print("Enter the alpha value(0.001):")
    alpha = float(input())#0.001
    
    print("Enter the eta value(0.001):")
    eta = float(input()) #0.001
    
    print("Enter the dec value(0.00001):")
    dec = float(input()) #0.00001
    
    train_x, train_y, test_x, test_y = load_data()
    cost, acc, y_pred = train(train_x, train_y, test_x, test_y, epochs, batch, alpha, eta, dec)

    x_a = [i for i in range(acc.shape[0])]
    x_c = [i for i in range(len(cost))]
    print('Final prediction accuracy is: ', acc[-1])
    
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(x_c, cost)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Cost")

    ax2 = fig.add_subplot(212)
    ax2.plot(x_a, acc)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Accuracy")
    
    plt.show()
    #plt.savefig('Cost-Accuracy.png', dpi=300)

    miscl_img = test_x[test_y != y_pred][:25]
    correct_lab = test_y[test_y != y_pred][:25]
    miscl_lab = y_pred[test_y != y_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) T: %d P: %d' % (i + 1, correct_lab[i], miscl_lab[i]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()