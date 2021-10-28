import numpy as np
import matplotlib.pyplot as plt
from implementations import *

import data_processing
import datetime
from core import *


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

### normalization on the input data (train/test)
data = data_processing.normalization(data)

# ### train/test split
all_x_train, x_test = data[:len(all_id_train)], data[:len(all_id_train)]

# split_fraction = 0.8
# (id_train, y_train, x_train), (id_eval, y_eval, x_eval), (x_test) = \
#     data_processing.data_split(all_id_train, all_y_train, data, split_fraction=split_fraction)
# print('data loading and splitting finish')

# Define the parameters of the algorithm.
epoch_num = 20
initial_gamma = 0.1
final_gamma = 0.001
gamma_decay = 0.5
batch_size = 2
# set the regularization term for initial feature as 0.1, and polynomial feature as 0.2
_lambda1 = 0.05
_lambda2 = 0.1
# _lambda = np.concatenate((np.ones(24)*_lambda1,np.ones(529)*_lambda2))
_lambda = 0.1



model_list = [
    'logistic regression',
    'reg logistic regression',
    'reg logistic regression with tricks',
]

k_fold = 5
print('all_x_train', len(all_x_train))
val_inds, train_inds = data_processing.cross_validation_kfold(all_x_train, num_folds=5)
print('val_id, train_id', val_inds, train_inds)

for model in model_list[1:]:

    predictions = []

    for val_ind, train_ind in zip(val_inds, train_inds):

        x_val, y_val = np.array(all_x_train)[np.array(val_ind)], np.array(all_y_train)[np.array(val_inds)]
        x_train, y_train = np.array(all_x_train)[np.array(train_ind)], np.array(all_y_train)[np.array(train_ind)]

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
        print("Best evaluation epoch: {idx} \nAccuracy: {acc}%". format(idx=np.argmax(accuracy_eval), acc=np.max(accuracy_eval)))

        predictions.append(np.max(accuracy_eval))

        ## TODO: visualization

    print(f'{model} average accuracy:', np.mean(predictions))

# # # Start test
# pred_test = np.round(sigmoid(x_test.dot(np.array(ws)[np.argmax(accuracy_eval),:])))
# pred_test[pred_test == 0] = -1
# submit_path = data_file + f"/submission_{model}.csv"
# data_loader.create_csv_submission(id_test, pred_test, submit_path)
