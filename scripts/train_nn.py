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

### normalization on the input data (train/test)
data = data_processing.normalization(data)

# ### train/test split
all_x_train, x_test = data[:len(all_id_train)], data[:len(all_id_train)]

k_fold = 5
print('all_x_train', len(all_x_train))
val_inds, train_inds = data_processing.cross_validation_kfold(all_x_train, num_folds=5)
print('val_id, train_id', val_inds, train_inds)

predictions = []

learning_rate = 0.1
epochs = 100
batch_size = 10

for val_ind, train_ind in zip(val_inds, train_inds):

    x_val, y_val = np.array(all_x_train)[np.array(val_ind)], np.array(all_y_train)[np.array(val_inds)]
    x_train, y_train = np.array(all_x_train)[np.array(train_ind)], np.array(all_y_train)[np.array(train_ind)]

    print('x_tra, y_t', x_train.shape, y_train.shape)
    x_train = x_train
    y_train = y_train.reshape(-1, 1)

    layer_size = [x_train.shape[1], int(x_train.shape[1]//2), 1]
    # Train our neural network!
    network = NeuralNetwork(x_train, y_train, layer_size, lr=learning_rate, n_epochs=epochs, batch_size=batch_size)
    network.train(x_train, y_train)
    eval_pred = network.predict(x_val)

    print('eval_pred', eval_pred.shape)
    eval_pred[eval_pred > 0.5] = 1
    eval_pred[eval_pred <= 0.5] = 0
    accuracy = network.accuracy_metric(y_val, eval_pred)
    print('accuracy', accuracy)

    predictions.append(accuracy)

    ## TODO: visualization

print(f'average accuracy:', np.mean(predictions))