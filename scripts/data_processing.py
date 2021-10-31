# -*- coding: utf-8 -*-
import csv
import numpy as np
import helper

np.random.seed(0)


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def load_data(data_path):
    '''
    Load training/test data.
    data_path: path of train/test.csv
    '''
    data = np.loadtxt(data_path, delimiter=',', skiprows=1, converters={1: lambda x: int(x == 's'.encode('utf-8'))})
    ids, y, input_data = data[:, 0], data[:, 1], data[:, 2:]

    return ids, y, input_data


def clean_data(data):
    # Replace -999 with mean value
    for i in range(data.shape[1]):
        data[data[:, i] == -999, i] = np.mean(data[data[:, i] > -999, i])
    return data


def feature_cleaning(data):
    # get rid of useless features
    chosen_feature = np.ones(30)
    black_list = [14, 15, 17, 18, 20, 25, 28]
    chosen_feature[black_list] = 0
    data = data[:,np.bool_(chosen_feature)]
    return data

def poly_feature_aug(data):
    # Genetate quadratic polynomial feature
    new_feature = np.reshape(np.tile(np.expand_dims(data, axis=2), (1, 1, data.shape[1])) *
                             np.tile(np.expand_dims(data, axis=1), (1, data.shape[1], 1)),
                             (data.shape[0], data.shape[1] ** 2))
    return new_feature


def standardize(x, nonlinear_trasform = True):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if nonlinear_trasform == True:
        std_x = np.std(x, axis=0)
        x = np.sign(x) * (std_x * np.log(np.abs(x/std_x)+1))
    std_x_new = np.std(x, axis=0)
    x = x / std_x_new
    return x


def normalization(data, new_feature = None):
    '''
        normalization function.
        data: initial data
        new_feature: polynomial feature
    '''

    if new_feature is None:
        data = np.concatenate((np.ones((data.shape[0],1)), standardize(data, nonlinear_trasform = True)), axis=1)
    else:
        data = np.concatenate((np.ones((data.shape[0], 1)), standardize(data, nonlinear_trasform = True), standardize(new_feature, nonlinear_trasform = True)), axis=1)
    return data


def data_split(ids, y, input, split_fraction=0.8):

    labeled_data_num = len(ids)
    #test split
    test_data = input[labeled_data_num:]

    ### train/val split
    split_number = int(labeled_data_num * split_fraction)
    train_id, train_label, train_data = ids[:split_number], y[:split_number], input[:split_number]
    eval_id, eval_label, eval_data = ids[split_number:], y[split_number:], input[split_number:labeled_data_num]

    return (train_id, train_label, train_data), (eval_id, eval_label, eval_data), (test_data)


# split data into k folds
def cross_validation_kfold(data, num_folds=5):
    '''
    K-fold cross-validation: divides all the samples in num_folds groups of samples
    '''
    train_id = []
    test_id = []
    data_index = np.arange(data.shape[0])

    # size of each folder
    fold_size = int(data.shape[0] / num_folds)

    for i in range(num_folds):

        test_fold = data_index[fold_size * i:fold_size * (i + 1)]
        train_fold = np.delete(data_index, test_fold)

        test_id.append(test_fold)
        train_id.append(train_fold)

    return test_id, train_id