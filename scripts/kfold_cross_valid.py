
# split data into k folds
def cross_validation__kfold(ids, num_folds=10):
    '''
    K-fold cross-validation: divides all the samples in num_folds groups of samples
    '''
    train_id = list()
    test_id = list()
    data_index = np.arange(ids.shape[0])
    
    # size of each folder
    fold_size = int(ids.shape[0] / num_folds)
    
    for i in range(num_folds):
        test_fold = list()
        train_fold = list()
        
        test_fold = data_index[fold_size*i:fold_size*(i+1)]
        train_fold = np.delete(data_index, test_fold)
        
        test_id.append(test_fold)
        train_id.append(train_fold)
    return test_id, train_id
