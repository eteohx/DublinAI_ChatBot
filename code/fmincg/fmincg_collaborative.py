# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:18:57 2020

Based on:
https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-8/
https://www.coursera.org/learn/machine-learning

@author: Administrator
"""
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

def cost(params, Y, R, num_features, learning_rate):
    Y = np.matrix(Y)  
    R = np.matrix(R)  
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    
    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)
    
    # initializations
    J = 0
    X_grad = np.zeros(X.shape)  # (1682, 10)
    Theta_grad = np.zeros(Theta.shape)  # (943, 10)
    
    # compute the cost
    error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
    squared_error = np.power(error, 2)  # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)
    
    # add the cost regularization
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))
    
    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
    
    return J, grad


data = loadmat('G:/My Drive/DublinAI/Mini Projects/chatbot/fmincg/ratings_small.mat')
Y = np.transpose(data['Y'])
R = np.transpose(data['R'])

movies = Y.shape[0]
users = Y.shape[1]
features = 15
learning_rate = 10.

X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

Ymean = np.zeros((movies, 1))
Ynorm = np.zeros((movies, users))

for i in range(movies):
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]

fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate), 
                method='CG', jac=True, options={'maxiter': 100})

X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

#np.savetxt("movie_features_fmincg.csv", X, delimiter=",")

predictions = X * Theta.T 
predictions_adjusted = np.zeros(predictions.shape)
for i in range(predictions.shape[1]):
    predictions_adjusted[:,i] = np.squeeze(predictions[:,i] + Ymean)

np.savetxt("predictions_adjusted.csv", predictions_adjusted, delimiter=",")
