# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:39:17 2018

@author: yang
"""

from utils import initialize,forward,compute_cost,backprop,update_p,load_data
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
def train(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False):#lr was 0.009
    np.random.seed(1)
    costs = []       
    parameters = initialize(layers_dims)    
    for i in range(0, num_iterations):
        AL, caches = forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backprop(AL, Y, caches)
        parameters = update_p(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()   
    return parameters  

def data_preprocessing():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    print ("The original shape of each input image is:{}".format(train_x_orig[0].shape))
    plt.figure(1)
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(train_x_orig[i])
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T 
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    X_train = train_x_flatten/255.
    X_test = test_x_flatten/255.
    print ("X_train's shape: " + str(X_train.shape))
    print ("X_test's shape: " + str(X_test.shape))
    dim_ori = train_x_orig.shape[1]*train_x_orig.shape[2]*train_x_orig.shape[3]
    return X_train, train_y, X_test, test_y,dim_ori

def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1,m))
    probas, caches = forward(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy: "  + str(np.sum((p == y)/m)))        
    return p

X_tr,y_tr,X_te,y_te,dim_ori = data_preprocessing()
layers_dims = [dim_ori, 20, 16, 10,1]
parameters = train(X_tr,y_tr, layers_dims,learning_rate=1e-2, num_iterations = 4000, print_cost = True)
result = predict(X_te,y_te,parameters)
