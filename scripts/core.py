import numpy as np
from implementations import *

def sigmoid(x):
    return 1/(1+np.exp(-x))

def train(model, y_train, x_train, w_initial, batch_size, max_iters, lambda_, gamma, final_gamma, gamma_decay):

    if model == 'least squares':
        w, loss = least_squares(y_train, x_train)

    elif model == 'least squares GD':
        w, loss = least_squares_GD(y_train, x_train, w_initial, max_iters, gamma)

    elif model == 'least squares SGD':
        w, loss = least_squares_GD(y_train, x_train, w_initial, max_iters, gamma)

    elif model == 'ridge regression':
        w, loss = ridge_regression(y_train, x_train, lambda_)

    elif model == 'logistic regression':
        w, loss = logistic_regression(y_train, x_train, w_initial, max_iters, gamma)

    elif model == 'reg logistic regression':
        w, loss = reg_logistic_regression(y_train, x_train, lambda_, w_initial, max_iters, gamma)

    elif model == 'reg logistic regression with tricks':
        w, loss = reg_logistic_regression_with_tricks(y_train, x_train, lambda_, w_initial,
                                                       batch_size, max_iters, gamma, final_gamma, gamma_decay)

    return w, loss


def evaluation(ws, x_eval, y_eval):

    if len(np.array(ws).shape)==1:
        ws = np.array(ws)
        pred_eval = np.round(sigmoid(x_eval.dot(ws.T)))
        accuracy_eval = np.mean(pred_eval == y_eval) * 100
    else:
        ws = np.array(ws)
        pred_eval = np.round(sigmoid(x_eval.dot(ws.T)))
        accuracy_eval = np.mean(pred_eval == np.tile(y_eval, (pred_eval.shape[1],1)).T, axis=0) * 100

    return accuracy_eval


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0