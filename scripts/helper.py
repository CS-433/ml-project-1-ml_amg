import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def batch_iter_improved(y, tx, batch_size, shuffle=True):
    """
    Generate a minibatch iterator for a dataset and interate through the whole epoch.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    start_index = 0
    while start_index < data_size:
        end_index = min(start_index + batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            start_index = end_index


def compute_loss(y, tx, w):
    '''compute the loss by MSE'''
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(tx))
    return mse

def compute_gradient_and_loss(y, tx, w):
    """Compute the gradient and loss"""
    e = y - tx.dot(w)
    grad = - np.transpose(tx).dot(e) / len(tx)
    loss = e.dot(e) / (2 * len(tx))
    return grad, loss

def sigmoid(x):
    """sigmoid function"""
    return 1/(1+np.exp(-x))

def compute_logistic_loss_and_grad(y, X, w):
    """Calculate the logistics loss and the gredient."""
    h_x = sigmoid(X.dot(w))
    logi_loss = np.mean(-y * np.log(h_x) - (1-y) * np.log(1-h_x+1e-6))
    grad = (X.T.dot(h_x-y))/y.shape[0]
    return logi_loss, grad

def compute_reg_logistic_loss_and_grad(y, X, w, lambda_):
    """Calculate the regularized logistics loss and the gredient."""
    h_x = sigmoid(X.dot(w))
    loss_term = np.mean(-y * np.log(h_x) - (1 - y) * np.log(1 - h_x + 1e-6))
    regu_term = np.sum(w ** 2) / (2 * y.shape[0])
    logi_loss = loss_term + lambda_ * regu_term
    grad = (X.T.dot(h_x - y) + lambda_ * w) / y.shape[0]
    return logi_loss, grad

def find_step_num(initial_gamma, final_gamma, gamma_decay, epoch_num):
    """find the step number to perform learning rate decay"""
    iter_num = 1
    gamma = initial_gamma
    while gamma > final_gamma:
        gamma = gamma * gamma_decay
        iter_num = iter_num +1
    step_num = int(epoch_num / iter_num)
    return step_num

def grid_search(initial_gamma_list, final_gamma_list, gamma_decay_list, _lambda_list):
    """generate grid search parameter matrix for learning rate and regularization weight"""
    dimensions = len(initial_gamma_list) * len(final_gamma_list) * len(gamma_decay_list) * len(_lambda_list)
    aa, bb, cc, dd = np.meshgrid(initial_gamma_list, final_gamma_list, gamma_decay_list, _lambda_list)
    aa, bb, cc, dd = np.reshape(aa, (dimensions, -1)), np.reshape(bb, (dimensions, -1)), \
                     np.reshape(cc, (dimensions, -1)), np.reshape(dd, (dimensions, -1))
    parameter_matrix = np.concatenate((aa, bb, cc, dd), axis=1)
    return parameter_matrix
