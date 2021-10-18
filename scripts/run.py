import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from logistic_regression import *
# from helper import *
import data_loader
import datetime

### data path
data_file = '/Users/zhoumu/Documents/code/ml-project-1/data'
train_path = data_file + "/train.csv"
test_path = data_file + "/test.csv"

### load the train / test data
all_train_id, all_train_label, all_train_data = data_loader.load_data(train_path)
test_id, test_label, test_data = data_loader.load_data(test_path)

### pre-processing
# TODO: clean the data / feature engineering

### normalization on the input data (train/test)
all_train_data, test_data = data_loader.normalization(all_train_data, test_data, concatenate=True)

### train/val split
split_fraction = 0.8
(train_id, train_label, train_data), (eval_id, eval_label, eval_data) = \
    data_loader.train_val_split(all_train_id, all_train_label, all_train_data, split_fraction=split_fraction)
print('data loading and splitting finish')

# Define the parameters of the algorithm.
epoch_num = 200
gamma = 0.001
lamda = 0.1
batch_size = 100

# Initialization
w_initial = np.zeros((train_data.shape[1]))

# Start training.
start_time = datetime.datetime.now()
sgd_losses, sgd_ws = stochastic_gradient_descent(
    train_label, train_data, w_initial, batch_size, epoch_num, lamda, gamma)
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("training time={t:.3f} seconds".format(t=exection_time))

# Start evaluation
sgd_ws = np.array(sgd_ws)
pred_eval = np.round(sigmoid(eval_data.dot(sgd_ws.T)))
accuracy_eval = np.mean(pred_eval == np.tile(eval_label,(pred_eval.shape[1],1)).T, axis=0) * 100
print("Best evaluation epoch: {idx} \nAccuracy: {acc}%". format(idx=np.argmax(accuracy_eval), acc=np.max(accuracy_eval)))

## TODO: visualization


# # Start test
# pred_test = np.round(sigmoid(test_data.dot(sgd_ws[np.argmax(accuracy_eval),:])))
# pred_test[pred_test == 0] = -1
# submit_path = data_file + "/submission_baseline.csv"
# create_csv_submission(test_id, pred_test, submit_path)
