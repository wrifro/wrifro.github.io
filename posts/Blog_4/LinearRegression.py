import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import minimize

class LinearRegression:
    
    def __init__(self):
        self.w = np.zeros(2)
        self.loss_history = []
        self.score_history = []
    
    def fit(self,X,y):
        X = self.pad(X)
        
        w_hat = np.linalg.inv(X.T@X)@X.T@y
        
        self.w = w_hat
        
    def fit_gradient(self,X,y,alpha,m_epochs):
        X = self.pad(X)
        P = X.T@X
        q = X.T@y
        
        for epoch in range(m_epochs):
            w_prev = self.w

            grad = 2 * (P@w_prev - q)
            
            w_new = w_prev - (alpha * grad)
            
            prev_loss = self.loss(X,y,w_prev)
            
            self.w = w_new
    
                
    def score(self,X,y):
        ## This method returns the score/accuracy of the y values predicted with the most recent weight vector
        ## I found it easier to think of y as -1 and 1, like we did in perceptron, rather than 1 and 0, so I
        ## converted it to an array of -1s and 1s here for my own convenience.
        y_ = 2*y-1
        return(np.mean((y_== (2*((X@self.w) > 0)-1))))

    ## All methods below were adapted/borrowed from the notes for the lecture on Gradient Descent.
    def predict(self,X, w):
        return (X@w)

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def loss(self,X, y, w):
        y_hat = self.predict(X,w)
        loss_all = -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))
        return loss_all.mean()
    
    def pad(self,X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)