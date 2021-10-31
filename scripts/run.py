import numpy as np
import matplotlib.pyplot as plt
from implementations import *

import data_processing
import datetime
from core import *
from neural_network import *


### data path
data_file = '/Users/zhoumu/Documents/code/ml-project-1/data'
train_path = data_file + "/train.csv"
test_path = data_file + "/test.csv"

### load the train / test data
all_id_train, all_y_train, all_x_train = data_processing.load_data(train_path)
id_test, y_test, x_test = data_processing.load_data(test_path)
data = np.concatenate((all_x_train,x_test), axis=0)

### pre-processing
data = data_processing.feature_cleaning(data)
data = data_processing.clean_data(data)
# new_feature = data_loader.poly_feature_aug(data)

# ### train/test split
all_x_train, x_test = data[:len(all_id_train)], data[len(all_id_train):]

### normalization on the input data (train/test)
all_x_train = data_processing.normalization(all_x_train)
x_test = data_processing.normalization(x_test)


# Define the parameters of the algorithm.
epoch_num = 50
initial_gamma = 0.1
final_gamma = 0.001
gamma_decay = 0.5
batch_size = 32
# set the regularization term for initial feature as 0.1, and polynomial feature as 0.2
_lambda1 = 0.05
_lambda2 = 0.1
# _lambda = np.concatenate((np.ones(24)*_lambda1,np.ones(529)*_lambda2))
_lambda = 0.1


model_list = [
    'least squares',
    'least squares GD',
    'least squares SGD',
    'ridge regression',
    'logistic regression',
    'reg logistic regression',
    'reg logistic regression with tricks',
    'nn',

]

k_fold = 5
val_inds, train_inds = data_processing.cross_validation_kfold(all_x_train, num_folds=5)

model_results = []

for model in model_list:

    predictions = []

    for val_ind, train_ind in zip(val_inds[:1], train_inds[:1]):

        x_val, y_val = np.array(all_x_train)[np.array(val_ind)], np.array(all_y_train)[np.array(val_ind)]
        x_train, y_train = np.array(all_x_train)[np.array(train_ind)], np.array(all_y_train)[np.array(train_ind)]

        if model != 'nn':
            # Initialization
            w_initial = np.zeros((x_train.shape[1]))

            # Start training.
            start_time = datetime.datetime.now()
            ws, losses = train(model, y_train, x_train, w_initial, batch_size, epoch_num, _lambda, initial_gamma, final_gamma, gamma_decay)
            end_time = datetime.datetime.now()
            exection_time = (end_time - start_time).total_seconds()
            print("Model {m}:training time={t:.3f} seconds".format(m=model, t=exection_time))

            # Start evaluation
            accuracy_eval = evaluation(ws, x_val, y_val)
            accuracy = np.max(accuracy_eval)
            print("Best evaluation epoch: {idx} \nAccuracy: {acc}%". format(idx=np.argmax(accuracy_eval), acc=accuracy))

            # # Start test
            if 'tricks' not in model:
                pred_test = np.round(sigmoid(x_test.dot(np.array(ws))))
            else:
                pred_test = np.round(sigmoid(x_test.dot(np.array(ws)[np.argmax(accuracy_eval), :])))

            pred_test[pred_test == 0] = -1

        else:

            y_train = y_train.reshape(-1, 1)

            layer_size = [x_train.shape[1], int(x_train.shape[1] * 2), 1]
            # Train our neural network
            network = NeuralNetwork(x_train, y_train, layer_size, lr=initial_gamma, n_epochs=epoch_num,
                                    batch_size=batch_size, learning_rate_decay=gamma_decay)
            network.train(x_train, y_train)
            eval_pred = network.predict(x_val)

            eval_pred[eval_pred > 0.5] = 1
            eval_pred[eval_pred <= 0.5] = 0
            accuracy = network.accuracy_metric(y_val, eval_pred)
            print('accuracy', accuracy)

            ### start test
            pred_test = network.predict(x_test)

        predictions.append(accuracy)

        ## TODO: visualization

    results = {}
    results['model'] = model
    results['accuracy'] = np.mean(predictions)
    results['epoch'] = epoch_num
    results['loss'] = losses
    print(f'{model} -- average accuracy after cross validation:', results['accuracy'])

model_results.append(results)
import json
with open(f'{data_file}/model_results.json', 'w') as f:
    json.dump(model_results, f)

submit_path = data_file + f"/submission_{model}.csv"
data_processing.create_csv_submission(id_test, pred_test, submit_path)
