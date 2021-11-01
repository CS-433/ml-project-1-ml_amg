import numpy as np
from helper import *

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
            loss, grad = compute_reg_logistic_loss_and_grad(batch_y, batch_X, w, lambda_)
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