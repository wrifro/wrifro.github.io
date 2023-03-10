import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import minimize

class LogisticRegression:
    
    def __init__(self):
        self.w = np.zeros(3)
        self.loss_history = []
        self.score_history = []
    
    def fit(self,X,y,alpha,m_epochs):
        ## The fit method uses gradient descent to calculate the best possible weight vector
        ## for a line dividing - or trying to divide – two sets of points. This algorithm 
        ## works by calculating a new gradient using the sigmoid function; when this function
        ## has reached a minimum, empirical risk has been minimized. So, at each step this method
        ## updates the weight vector (self.w) by subtracting the newly calculated gradient * the learning
        ## rate alpha from the old weight, until loss has been minimized.
        
        p = X.shape[1] #-number of features
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        self.w = (np.random.rand(p+1))
        self.loss_history.append(self.loss(X_,y,self.w))
        self.score_history.append(self.score(X_,y))
        
        for epoch in range(m_epochs):
            w_prev = self.w
            
            gradient = self.gradient(X_,y,w_prev)
            
            w_new = w_prev - (alpha * gradient)
            
            prev_loss = self.loss(X_,y,w_prev)
            
            self.w = w_new
            
            new_loss = self.loss(X_,y,w_new)
            self.loss_history.append(new_loss)
            
            self.score_history.append(self.score(X_,y))

            if np.isclose(new_loss, prev_loss):          
                print("converged")
                break
    
    def fit_stochastic(self,X,y,m_epochs,batch_size,alpha):
        ## Stochastic fit also uses gradient descent to calculate the best possible weight vector.
        ## Unlike regular gradient descent, however, stochastic gradient descent does not consider
        ## every point at once. Instead, this algorithm randomly divides the points into subsets
        ## (the size of which are specified by the "batch_size" parameter). It then recalculates 
        ## the weight vector for each batch one at a time; only after every batch has been cycled
        ## through – this corresponds to one epoch – is the loss updated and compared to the previous loss.
        ## The function's code is quite similar to the fit method; the only difference is that 
        ## weight is updated for each batch rather than for all points at once. 
        
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        n = X_.shape[0]
        p = X.shape[1] #-number of features

        self.w = (np.random.rand(p+1))
        self.loss_history.append(self.loss(X_,y,self.w))
        self.score_history.append(self.score(X_,y))
        prev_loss = self.loss(X_,y,self.w)
        
        for j in np.arange(m_epochs):
            order = np.arange(n)
            np.random.shuffle(order)
            
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]

                grad = self.gradient(x_batch, y_batch,self.w)
                self.w -= (alpha * grad)
                
            new_loss = self.loss(X_,y,self.w)
            self.loss_history.append(self.loss(X_,y,self.w))
            self.score_history.append(self.score(X_,y))
            
            if np.isclose(new_loss, prev_loss):          
                print("converged")
                break
            else:
                prev_loss = new_loss
           


                
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
    
    def gradient(self,X,y,w):
        grad = np.multiply( ( self.sigmoid (X@self.w) - y)[:,np.newaxis], X ).mean(axis=0)
        return grad