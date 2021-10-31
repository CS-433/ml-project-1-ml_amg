## cs433ML-project 1

In this repository, we provide the code for Project1 of the cs433 Machine Learning course.

### Dataset
The trainining dataset contains 250,000 data with signal/background labels, and the test dataset contains 568,238 data without labels. As input we have 30-dimensional space of features. Each vector of features represents the decay signature of a collision event, and our goal is to predict whether this event is signal (a Higgs boson) or background (something else).
You can download the dataset from the [ML Higgs competetion](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) from AIcrowd.

### Code
You will find the code in ./scripts folder.

### How to run the models
`run.py`: you can run all the models in the report by using `python run.py`, it will print the mean accuracy on validation data and generate the CSV files of test data from each model.

### The code structure
#### `scripts/data_processing.py`
It contains all the data pre-processing and post-processing methods:
- load data: **load_data(data_path)**
- clean data: **clean_data(data)**
- feature engineering: **feature_cleaning(data)**
- data standardize: **poly_feature_aug(data)**
- data normalization: **standardize(x, nonlinear_trasform)**
- k-fold cross validation: **normalization(data, new_feature)**
- create csv submission file: **create_csv_submission(ids, y_pred, name)**

#### `scripts/helper.py`
It contains all the tool functions we need in the project.
- basic generate minibatch iterator (from exercise): **batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)**
- the improved function for generating minibatch iterator: **(y, tx, batch_size, shuffle=True)**
- compute mse loss: **compute_loss(y, tx, w)**
- comupte the gradient and the mse loss: **compute_gradient_and_loss(y, tx, w)**
- compute the logistic loss and the gradient: **compute_logistic_loss_and_grad(y, X, w)**
- compute the regularized logistic loss and the gradient: **compute_reg_logistic_loss_and_grad(y, X, w, lambda_)**
- the improved function for computing the regularized logistic loss and the gradient: **compute_improv_reg_logistic_loss_and_grad(y, X, w, lambda_)**
- sigmoid function: **sigmoid(x)**
- function to find the step_size: **find_step_num(initial_gamma, final_gamma, gamma_decay, epoch_num)**

#### `scripts/implementations.py`
It contains the 6 functions in project1 description
- linear regression using gradient descent: **least_squares_GD(y, tx, initial_w, max_iters, gamma)**
- linear regression using stochastic gradient descent: **least_squares_SGD(y, tx, initial_w, max_iters, gamma)**
- least squares regression using normal equations: **least_squares(y, tx)**
- ridge regression using normal equations: **ridge_regression(y, tx, lambda_)**
- logistic regression using gradient descent or SGD: **logistic_regression(y, tx, initial_w, max_iters, gamma)**
- regularized logistic regression using gradient descent or SGD: **reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)**

#### `scripts/reg_logistic_regression_with_tricks.py`
It contains the improved function with several tricks for the regularized logistic regression using gradient descent or SGD.
- regularized logistic regression using gradient descent or SGD with tricks: **reg_logistic_regression_with_tricks(y, tx, lambda_, initial_w, batch_size, epoch_num, initial_gamma, final_gamma, gamma_decay)**

#### `scripts/neural_network.py`
It contains the implementation of the neural network, where we use ReLU for hidden layer and Sigmoid for the last layer.

#### `scripts/core.py`
It contains the core function for training and evaluation.
- train the different models: **train(model, y_train, x_train, w_initial, batch_size, max_iters, lambda_, gamma, final_gamma, gamma_decay)**
- evaluate the model: **evaluation(ws, x_eval, y_eval)**

#### `scripts/run.py`
It contains the running script for all the models.

#### `scripts/plots.py`
It contains the visualization code.

