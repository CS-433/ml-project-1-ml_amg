import numpy as np
import matplotlib.pyplot as plt
from implementations import *
# from helper import *
import data_loader
import datetime
from core import *

### data path
data_file = 'D:/Courses/Machine learning/Project1/data'
train_path = data_file + "/train.csv"
test_path = data_file + "/test.csv"

### load the train / test data
all_id_train, all_y_train, all_x_train = data_loader.load_data(train_path)
id_test, y_test, x_test = data_loader.load_data(test_path)
data = np.concatenate((all_x_train,x_test), axis=0)
### pre-processing
data = data_loader.feature_cleaning(data)
data = data_loader.clean_data(data)
new_feature = data_loader.poly_feature_aug(data)

### normalization on the input data (train/test)
data = data_loader.normalization(data, new_feature)

### train/val split
split_fraction = 0.8
(id_train, y_train, x_train), (id_eval, y_eval, x_eval), (x_test) = \
    data_loader.data_split(all_id_train, all_y_train, data, split_fraction=split_fraction)
print('data loading and splitting finish')

# Define the parameters of the algorithm.
epoch_num = 200
initial_gamma = 0.1
gamma_decay = 0.5
batch_size = 200
# set the regularization term for initial feature as 0.1, and polynomial feature as 0.2
_lambda1 = 0.05
_lambda2 = 0.1
_lambda = np.concatenate((np.ones(24)*_lambda1,np.ones(529)*_lambda2))
# _lambda = 0.1

# Initialization
w_initial = np.zeros((x_train.shape[1]))

# Start training.
start_time = datetime.datetime.now()
model = 'reg logistic regression with tricks' # TODO: CHANGE THE MODEL NAME
losses, ws = train(model, y_train, x_train, w_initial, batch_size, epoch_num, _lambda, initial_gamma, gamma_decay)
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("Model {m}:training time={t:.3f} seconds".format(m=model, t=exection_time))

# Start evaluation
accuracy_eval = evaluation(ws, x_eval, y_eval)
print("Best evaluation epoch: {idx} \nAccuracy: {acc}%". format(idx=np.argmax(accuracy_eval), acc=np.max(accuracy_eval)))


## TODO: visualization


# # Start test
pred_test = np.round(sigmoid(x_test.dot(np.array(ws)[np.argmax(accuracy_eval),:])))
pred_test[pred_test == 0] = -1
submit_path = data_file + "/submission_newcode_weight.csv"
data_loader.create_csv_submission(id_test, pred_test, submit_path)
