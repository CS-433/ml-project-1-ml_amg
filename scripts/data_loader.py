# -*- coding: utf-8 -*-
import csv
import numpy as np
import helper

np.random.seed(0)

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


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
    # TODO: clean the data

    return None


def pre_processing(data):
    # TODO: feature engineering

    return None


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def normalization(train, test, concatenate=False):
    '''
        normalization function.
        train: list, train set
        test: list, test set
        concatenate: bool, do normalization on train and test set together or not.
    '''

    if concatenate:
        ### normalize train and test set together
        train_test_data = np.concatenate((train, test), axis=0)
        train_test_data, _, _ = standardize(train_test_data)
        train_test_data = np.c_[np.ones(train_test_data.shape[0]), train_test_data]
        train, test = train_test_data[:train.shape[0], :], train_test_data[train.shape[0]:, :]

    else:
        train, _, _ = standardize(train)
        test, _, _  = standardize(test)
        train = np.c_[np.ones(train.shape[0]), train]
        test = np.c_[np.ones(test.shape[0]), test]

    return train, test


def train_val_split(ids, y, input, split_fraction=0.8):

    ### train/val split
    split_number = int(ids.shape[0] * split_fraction)
    train_id, train_label, train_data = ids[:split_number], y[:split_number], input[:split_number]
    eval_id, eval_label, eval_data = ids[split_number:], y[split_number:], input[split_number:]

    return (train_id, train_label, train_data), (eval_id, eval_label, eval_data)
