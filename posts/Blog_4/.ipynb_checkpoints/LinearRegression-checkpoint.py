import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import minimize

class LinearRegression:
    
    def __init__(self):
        self.loss_history = []
        self.score_history = []
    
    def fit(self,X,y): # Main fit method - uses formula to calculate optimal w_hat
        X = self.pad(X)
        self.X = X
        
        w_hat = np.linalg.inv(X.T@X)@X.T@y
        
        self.w = w_hat
        
    def fit_gradient(self,X,y,alpha,m_epochs): # Gradient Descent fit method - calculates the gradient
        # and updates for each iteration in m_epochs. Speed of gradient descent is dictated by alpha,
        # the learning rate for the function.
        
        X = self.pad(X)
        self.X = X
        
        self.w = np.zeros(X.shape[1])
        
        P = (X.T)@X
        q = (X.T)@y
        
        for epoch in range(m_epochs):
            w_prev = self.w

            grad = 2 * (P@self.w - q)
            
            w_new = w_prev - (alpha * grad)
            
            self.w = w_new
            
            score = self.score(X,y)
            
            self.score_history.append(score)
            
    def fit_stochastic(self,X,y,alpha, m_epochs,batch_size):
        # Runs much like regular gradient descent, but breaks up datapoints into smaller batches
        # to recalculate the gradient several times in each step. Batch_size dictates how many
        # datapoints are in each batch.
        
        X = self.pad(X)
        n = X.shape[0]
        self.w = np.zeros(X.shape[1])
        
        P = (X.T)@X
        q = (X.T)@y
        
        for j in np.arange(m_epochs):
            order = np.arange(n)
            np.random.shuffle(order)
            
            for batch in np.array_split(order, n // batch_size + 1):
                w_prev = self.w
                
                x_batch = X[batch,:]
                y_batch = y[batch]

                grad = 2 * (P@self.w - q)
            
                w_new = w_prev - (alpha * grad)
            
                self.w = w_new
                
            score = self.score(X,y)
            self.score_history.append(score)
    
                
    def score(self,X,y):
        # Calculates score/accuracy for the model based on the parameters used to fit it.
        
        y_hat = self.predict(X,self.w)
        numerator = np.sum((y_hat - y)**2)
        y_bar = np.mean(y)
        denominator = np.sum((y_bar - y)**2)
        
        return(1 - (numerator/denominator) )
        

    ## All methods below were adapted/borrowed from the notes for the lecture on Gradient Descent.
    def predict(self,X, w):
        return (X@w)
    
    def pad(self,X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)