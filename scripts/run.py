import numpy as np
import matplotlib.pyplot as plt
from implementations import *
# from helper import *
import data_loader
import datetime
from core import *

### data path
data_file = '/Users/zhoumu/Documents/code/ml-project-1/data'
train_path = data_file + "/train.csv"
test_path = data_file + "/test.csv"

### load the train / test data
all_id_train, all_y_train, all_x_train = data_loader.load_data(train_path)
id_test, y_test, x_test = data_loader.load_data(test_path)

### pre-processing
# TODO: clean the data / feature engineering

### normalization on the input data (train/test)
all_x_train, x_test = data_loader.normalization(all_x_train, x_test, concatenate=True)

### train/val split
split_fraction = 0.8
(id_train, y_train, x_train), (id_eval, y_eval, x_eval) = \
    data_loader.train_val_split(all_id_train, all_y_train, all_x_train, split_fraction=split_fraction)
print('data loading and splitting finish')

# Define the parameters of the algorithm.
epoch_num = 200
gamma = 0.001
lamda = 0.1
batch_size = 100

# Initialization
w_initial = np.zeros((x_train.shape[1]))

# Start training.
start_time = datetime.datetime.now()
model = 'SGD' # TODO: CHANGE THE MODEL NAME
loss, ws = train(model, y_train, x_train, w_initial, batch_size, epoch_num, lamda, gamma)
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("training time={t:.3f} seconds".format(t=exection_time))

# Start evaluation
accuracy_eval = evaluation(ws, x_eval, y_eval)
print("Best evaluation epoch: {idx} \nAccuracy: {acc}%". format(idx=np.argmax(accuracy_eval), acc=np.max(accuracy_eval)))


## TODO: visualization


# # Start test
# pred_test = np.round(sigmoid(x_test.dot(sgd_ws[np.argmax(accuracy_eval),:])))
# pred_test[pred_test == 0] = -1
# submit_path = data_file + "/submission_baseline.csv"
# create_csv_submission(id_test, pred_test, submit_path)
