import numpy as np
from implementations import *

def sigmoid(x):
    return 1/(1+np.exp(-x))

def train(model, y_train, x_train, w_initial, batch_size, max_iters, lambda_, gamma, final_gamma, gamma_decay):

    if model == 'least squares':
        loss, ws = least_squares(y_train, x_train)

    elif model == 'least squares GD':
        loss, ws = least_squares_GD(y_train, x_train, w_initial, max_iters, gamma)

    elif model == 'least squares SGD':
        loss, ws = least_squares_GD(y_train, x_train, w_initial, max_iters, gamma)

    elif model == 'ridge regression':
        loss, ws = ridge_regression(y_train, x_train, lambda_)

    elif model == 'logistic regression':
        loss, ws = logistic_regression(y_train, x_train, w_initial, max_iters, gamma)

    elif model == 'reg logistic regression':
        loss, ws = reg_logistic_regression(y_train, x_train, lambda_, w_initial, max_iters, gamma)

    elif model == 'reg logistic regression with tricks':
        loss, ws = reg_logistic_regression_with_tricks(y_train, x_train, lambda_, w_initial,
                                                       batch_size, max_iters, gamma, final_gamma, gamma_decay)

    return loss, ws


def evaluation(ws, x_eval, y_eval):

    ws = np.array(ws)
    pred_eval = np.round(sigmoid(x_eval.dot(ws.T)))
    accuracy_eval = np.mean(pred_eval == np.tile(y_eval, (pred_eval.shape[1],1)).T, axis=0) * 100

    return accuracy_eval