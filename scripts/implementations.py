import numpy as np
from helper import *

### Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w, loss = None
    return w, loss

### Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w, loss = None
    return w, loss


### Least squares regression using normal equations
def least_squares(y, tx):
    w = np.linalg.inv(tx.transpose().dot(tx)).dot(tx.transpose().dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

### Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    I = np.identity(tx.shape[1])
    n = tx.shape[0]
    lambda_prime = 2 * n * lambda_
    w = np.linalg.inv(tx.transpose().dot(tx) + lambda_prime.dot(I)).dot(tx.transpose().dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

### Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w, loss = None
    return w, loss

### Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = loss = None
    return w, loss

