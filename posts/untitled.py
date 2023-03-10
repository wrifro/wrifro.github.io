import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class Perceptron:
    
    def __init__(self):
        self.w = []
        self.history = []
    
    def fit(self,X,y,max_steps):
        ## The fit method takes in a feature matrix X, and a classification matrix y, the latter of which corresponds to the
        ## grouping of the features of X. Starting with a random weight and improving the weight based on
        ## its predictions for classifications of various random points, the fit function gradually alters
        ## the weight variable until 100% accuracy is achieved (i.e. all points are correctly classified
        ## according to the prediction) or until the program runs for a user-specified maximum number of steps.
        
        n = X.shape[0] #-number of values
        p = X.shape[1] #-number of features
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        y_ = y = 2*(y) - 1 # target vector - make it equal to -1s and 1s rather than 0s and 1s

        ## Initialize random weight vector w_0
        w_i = (np.random.rand(p+1))
        self.w = w_i

        ## Until termination:
        # - pick random index i
        # - compute the fxn
        for step in range(max_steps):
            #self.w.append(w_i)
            w = w_i

            i = np.random.randint(0, high=n-1, dtype=int)
            
            X_i = X_[i]
            y_i = y_[i]

            y_hat = np.dot(w_i,X_i)
            
            w_i = ( (np.multiply(y_hat, y_i) >= 0) * w_i)  + ( (np.multiply(y_hat, y_i) < 0) * (w_i + np.multiply(y_i,X_i)) )
            self.w = w_i
            
            score = self.score(X,y_)
            self.history.append(score)
            if score == 1:
                break
                
    def predict(self,X):
        ## Given the most up-to-date weight vector, this method returns the predicted y values for each item in the feature matrix
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        w_i = self.w
        
        y_hat = (2 * (np.dot(X_,w_i) > 0) - 1)
        return(y_hat)
    
    def score(self,X,y):
        ## This method returns the score/accuracy of the y values predicted with the most recent weight vector
        return(np.mean(y==self.predict(X)))
    
    def update(self,X,y,w):
        ## Same as fit method, but updates only one step at a time
        ## This allows user to see change from step to step
        
        n = X.shape[0] #-number of values
        p = X.shape[1] #-number of features
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        y_ = y = 2*(y) - 1 # target vector - make it equal to -1s and 1s rather than 0s and 1s

        i = np.random.randint(0, high=n-1, dtype=int)

        X_i = X_[i]
        y_i = y_[i]

        y_hat = np.dot(w,X_i)

        w_next = ( (np.multiply(y_hat, y_i) >= 0) * w)  + ( (np.multiply(y_hat, y_i) < 0) * (w + np.multiply(y_i,X_i)) )
        self.w = w_next

        return (w_next, self.score(X,y),i)
