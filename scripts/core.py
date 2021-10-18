import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def stochastic_gradient_descent(y_train, x_train, w_initial, batch_size, epoch_num, lamda, gamma):
    # TODO
    loss, ws = None
    return loss, ws

def train(model, y_train, x_train, w_initial, batch_size, epoch_num, lamda, gamma):

    if model == 'SGD':
        model_func = stochastic_gradient_descent

    ### TODO: add the other models
    # else:

    loss, ws = model_func(y_train, x_train, w_initial, batch_size, epoch_num, lamda, gamma)
    return loss, ws


def evaluation(ws, x_eval, y_eval):

    ws = np.array(ws)
    pred_eval = np.round(sigmoid(x_eval.dot(ws.T)))
    accuracy_eval = np.mean(pred_eval == np.tile(y_eval,(pred_eval.shape[1],1)).T, axis=0) * 100

    return accuracy_eval

