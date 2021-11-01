import numpy as np
from implementations import *
import data_processing
import datetime
from core import *
from neural_network import *

# code parameters
data_file = 'D:/Courses/Machine learning/Project1/data'
use_grid_search = False
use_cross_validation = True
use_PCA = False
PCA_num = 100
use_poly_feature = False
model_list = [
    # 'logistic regression',
    # 'reg logistic regression',
    # 'reg logistic regression with tricks',
    'nn',
]
epoch_num = 50
batch_size = 200
k_fold = 5

np.random.seed(1)
### data path
train_path = data_file + "/train.csv"
test_path = data_file + "/test.csv"

### load the train / test data
all_id_train, all_y_train, all_x_train = data_processing.load_data(train_path)
id_test, y_test, x_test = data_processing.load_data(test_path)
data = np.concatenate((all_x_train,x_test), axis=0)

### pre-processing
data = data_processing.feature_cleaning(data)
data = data_processing.clean_data(data)
if use_poly_feature:
    new_feature = data_processing.poly_feature_aug(data)
    data = np.concatenate((data, new_feature), axis=1)
if use_PCA:
    data = data_processing.pca_decomposition(data, PCA_num)

### train/test split
all_x_train, x_test = data[:len(all_id_train)], data[len(all_id_train):]

### normalization on the input data (train/test)
all_x_train, mean_train, std_train, std_train_new = data_processing.normalization_train(all_x_train)
x_test = data_processing.normalization_test(x_test, mean_train, std_train, std_train_new)


#Grid search parameter generation
# initial_gamma_list = [0.05, 0.1, 0.2]
# final_gamma_list = [0.0005,0.001]
# gamma_decay_list = [0.3,0.5,0.7]
# _lambda_list = [0.05,0.1,0.2]

if use_grid_search:
    initial_gamma_list = [0.1]
    final_gamma_list = [0.0005,0.001]
    gamma_decay_list = [0.5]
    _lambda_list = [0.1]
    parameter_matrix = grid_search(initial_gamma_list, final_gamma_list, gamma_decay_list, _lambda_list)
else:
    parameter_matrix = np.array([[0.1,0.0005,0.5,0.1]])

val_inds, train_inds = data_processing.cross_validation_kfold(all_x_train, num_folds=5)

if use_cross_validation:
    fold = k_fold
else:
    fold = 1

model_results = []

for model in model_list:
    predictions_para = []
    pred_tests_para = []
    for i in range(parameter_matrix.shape[0]):
        initial_gamma, final_gamma, gamma_decay, _lambda = parameter_matrix[i,:]
        predictions = []
        pred_tests = []
        for val_ind, train_ind in zip(val_inds[:fold], train_inds[:fold]):

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
                    pred_test = sigmoid(x_test.dot(np.array(ws)))
                else:
                    pred_test = sigmoid(x_test.dot(np.array(ws)[np.argmax(accuracy_eval), :]))


            else:

                y_train = y_train.reshape(-1, 1)

                layer_size = [x_train.shape[1], int(x_train.shape[1] * 2), 1]
                # Train our neural network
                network = NeuralNetwork(x_train, y_train, layer_size, lr=initial_gamma, final_lr=final_gamma, n_epochs=epoch_num,
                                        batch_size=batch_size, learning_rate_decay=gamma_decay)
                losses = network.train(x_train, y_train)
                eval_pred = network.predict(x_val)

                eval_pred[eval_pred > 0.5] = 1
                eval_pred[eval_pred <= 0.5] = 0
                accuracy = network.accuracy_metric(y_val, eval_pred)
                print('accuracy', accuracy)

                ### start test
                pred_test = network.predict(x_test)

            predictions.append(accuracy)
            pred_tests.append(pred_test)
        predictions_para.append(np.mean(predictions))
        pred_tests_para.append(np.round(np.mean(np.array(pred_tests), axis=0)))

            ## TODO: visualization
    max_idx = np.argmax(predictions_para)
    final_pred_test = pred_tests_para[max_idx]
    final_pred_test[final_pred_test == 0] = -1
    initial_gamma, final_gamma, gamma_decay, _lambda = parameter_matrix[max_idx,:]
    results = {}
    results['model'] = model
    results['accuracy'] = predictions_para[max_idx]
    results['epoch'] = epoch_num
    results['loss'] = losses
    results['initial_gamma'], results['final_gamma'], results['gamma_decay'], results['_lambda'] = initial_gamma, final_gamma, gamma_decay, _lambda
    print(f'{model} -- average accuracy after cross validation:', results['accuracy'])
    print(f'Best training parameters initial_gamma: {initial_gamma}, final_gamma: {final_gamma}, gamma_decay: {gamma_decay}, _lambda: {_lambda}')

model_results.append(results)
import json
with open(f'{data_file}/model_results.json', 'w') as f:
    json.dump(model_results, f)

submit_path = data_file + f"/submission_{model}.csv"
data_processing.create_csv_submission(id_test, final_pred_test, submit_path)