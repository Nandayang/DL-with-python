# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:35:53 2018

@author: yang
"""
import numpy as np
import h5py
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def initialize(layer_dim,mode="xavier"):
    np.random.seed(3)
    parameters = {}
    for i in range(1,len(layer_dim)):
        if mode == "he":           
            parameters["W"+str(i)] = np.random.randn(layer_dim[i],layer_dim[i-1])*np.sqrt(2.0/layer_dim[i-1])
        elif mode == "xavier":
            parameters["W"+str(i)] = np.random.randn(layer_dim[i],layer_dim[i-1])*np.sqrt(1.0/layer_dim[i-1])
        elif mode == "benjio_initial":
            parameters["W"+str(i)] = np.random.randn(layer_dim[i],layer_dim[i-1])*np.sqrt(2.0/(layer_dim[i-1]+layer_dim[i]))
        parameters["b"+str(i)] = np.zeros((layer_dim[i],1))
    assert(parameters['W' + str(i)].shape == (layer_dim[i], layer_dim[i-1]))
    assert(parameters['b' + str(i)].shape == (layer_dim[i], 1))
    print("The model has {} layers,W1 to".format(len(layer_dim))+" W"+str(i)+" has been initialized!")
    return parameters

def forward_sinle(A_prev, W, b, activation="relu",kp=1):
    Z = np.dot(W,A_prev)+b
    linear_cache = (A_prev, W, b)
    if activation == "sigmoid":
        A = 1/(1+np.exp(-Z))
        activation_cache = Z
    else:        
        A = np.maximum(0,Z)
        activation_cache = Z
    D = np.random.rand(A.shape[0],A.shape[1])
    D = D < kp
    A = np.multiply(A,D)
    A /= kp     
    cache_all = (linear_cache,activation_cache,D)
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    return A,cache_all
        
def forward(X,parameters,keep_prob):
    "we are using relu as the activation function except the last layer"
    depth = len(parameters) // 2
    caches = []
    A = X
    for i in range(1,depth):
        A_prev = A
        A,cache = forward_sinle(A_prev, parameters['W'+str(i)], parameters['b'+str(i)], activation="relu",kp=keep_prob)
        caches.append(cache)
    A_last,cache = forward_sinle(A, parameters['W'+str(depth)], parameters['b'+str(depth)], activation="sigmoid",kp=1)
    caches.append(cache)
    assert(A_last.shape == (1,X.shape[1]))
    return A_last, caches

def compute_cost(A_last,Y,num_layer,lambd,parameters=None,add_reg=False):
    m = Y.shape[1]
    l = num_layer
    ce_cost = -1/m*np.sum(Y*np.log(A_last)+(1-Y)*np.log(1-A_last))
    if add_reg:
        reg_cost = 0
        for i in range(1,l):
            reg_cost += np.sum(np.square(parameters["W"+str(l)]))*lambd/(2*m)
        cost = ce_cost + reg_cost
    else:
        cost = ce_cost
    cost = np.squeeze(cost)
    return cost
        
def linear_back(dZ,cache,lambd):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dA_prev = np.dot(W.T,dZ)
    dW = 1/m*np.dot(dZ,A_prev.T) + lambd/m*W
    db = 1/m*np.sum(dZ,axis=1,keepdims=True)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev,dW,db

def activation_back(dA,cache,activation,D,keep_p=1):
    "we've known dA,Z and want to get dZ"
    Z = cache
    dA = np.multiply(dA,D)
    dA /= keep_p
    if activation == "sigmoid":
        s = 1/(1+np.exp(-Z))
        dZ = dA*s*(1-s)
    elif activation == "relu":
        dZ = np.array(dA,copy=True)
        dZ[Z<=0] = 0
    elif activation == "tanh":
        s = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        dZ = dA*(1-np.power(s,2))
    else:
        print ("Your activation is :"+str(activation))
        print ("please confirm your activation function to be one of the relu,tanh or sigmoid")
    assert (dZ.shape == Z.shape)
    return dZ

def linear_act_back(dA,cache,activation,lambd=0,add_reg=False,kp=1):
    linear_cache,activation_cache,D_cache = cache
#    print (activation)
    dZ = activation_back(dA,activation_cache,activation,D_cache,keep_p=kp)
    dA_prev,dW,db = linear_back(dZ,linear_cache,lambd)
    return dA_prev,dW,db

def backprop(A_last,Y,caches,lambd, middle_layer_activation, last_activation,add_reg=False,keep_prob=1):
    '''
    we assume the loss is : -1/m*sum(y*logA+(1-y)log(1-A)),so we need to get dA_last.
    we could derivate that the dA_last = -y/A+(1-y)/(1-a)
    '''

    L = len(caches)
    grads = {}
    Y = Y.reshape(A_last.shape)
    dA_last = -np.divide(Y,A_last)+np.divide(1-Y,1-A_last)
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_act_back(dA_last,current_cache,
                                                                                       activation=last_activation,
                                                                                       lambd=lambd,add_reg=add_reg,kp=1)                                                                                        
    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_temp, dW_temp, db_temp = linear_act_back(grads["dA" + str(i+2)],
                                                               current_cache,activation="relu",
                                                               lambd=lambd,
                                                               add_reg=add_reg,
                                                               kp=keep_prob)
        grads["dA" + str(i+1)] = dA_prev_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp
    return grads
    
def update_p(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    """
    
    L = len(parameters) // 2 
    for i in range(L):
        parameters["W" + str(i+1)] -= learning_rate*grads["dW" + str(i+1)]
        parameters["b" + str(i+1)] -= learning_rate*grads["db" + str(i+1)]
    return parameters    