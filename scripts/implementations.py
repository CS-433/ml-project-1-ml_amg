import numpy as np
from helper import *

### Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        epoch_losses = []
        # compute gradient and loss
        grad, loss = compute_gradient_and_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        epoch_losses.append(loss)
        print("Epoch ({bi}/{ti}): loss={l}".format(bi=n_iter+1, ti=max_iters, l=np.mean(epoch_losses)))

    return w, loss

### Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # implement stochastic gradient descent.
    w = initial_w
    batch_size = 1
    epoch_losses = []
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size):
            # compute gradient and loss
            grad, loss = compute_gradient_and_loss(batch_y, batch_tx, w)
            # update w by gradient
            w = w - gamma * grad
            epoch_losses.append(loss)
        print("Epoch ({bi}/{ti}): loss={l}".format(bi=n_iter+1, ti=max_iters, l=np.mean(epoch_losses)))

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
    w = np.linalg.inv(tx.transpose().dot(tx) + lambda_prime*I).dot(tx.transpose().dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


### Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for i in range(max_iters):
        loss, grad = compute_logistic_loss_and_grad(y, tx, w)
        w = w - gamma * grad
        # store w and loss
        losses.append(loss)
        ws.append(w)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=i, ti=max_iters - 1, l=loss))

    return w, loss

### Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for i in range(max_iters):
        loss, grad = compute_reg_logistic_loss_and_grad(y, tx, w, lambda_)
        w = w - gamma * grad
        # store w and loss
        losses.append(loss)
        ws.append(w)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=i, ti=max_iters - 1, l=loss))

    return w, loss

def reg_logistic_regression_with_tricks(y, tx, lambda_, initial_w, batch_size, epoch_num, initial_gamma, final_gamma, gamma_decay):
    ws = [initial_w]
    gamma = initial_gamma
    losses = []
    epoch_losses = []
    epoch_w = []
    w = initial_w
    step_num = find_step_num(initial_gamma, final_gamma, gamma_decay, epoch_num)
    for i in range(epoch_num):
        for batch_y, batch_X in batch_iter_improved(y, tx, batch_size, shuffle=True):
            loss, grad = compute_improv_reg_logistic_loss_and_grad(batch_y, batch_X, w, lambda_)
            w = w - gamma * grad
            # store w and loss
            losses.append(loss)
            epoch_losses.append(loss)
            epoch_w.append(w)
        print("Epoch ({bi}/{ti}): loss={l}".format(bi=i + 1, ti=epoch_num, l=np.mean(epoch_losses)))
        ws.append(np.mean(np.array(epoch_w), axis=0))
        epoch_losses = []
        epoch_w = []
        if i % step_num == 0:
            gamma = gamma * gamma_decay

    return ws, losses
