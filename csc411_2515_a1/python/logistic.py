""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid
import math

def mod_data(data):
    (ndata, dim) = data.shape
    one_array = np.ones((ndata,1))
    # modify data to become N x (M + 1) matrix
    # where the last element of every sample is 1
    return np.concatenate((data, one_array), axis=1)

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    data = mod_data(data)
    z_array = get_z_array(weights, data)
    return sigmoid(z_array)

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of binary targets. Values should be either 0 or 1
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy.  CE(p, q) = E_p[-log q].  Here
                       we want to compute CE(targets, y).
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    n_correct = 0
    n_total = len(targets)

    for i in xrange(len(targets)):
        target = targets[i][0]
        predict = y[i][0]
        if predict >= 0.5 and target == 0:
            n_correct += 1
        elif predict < 0.5 and target == 1:
            n_correct += 1
    frac_correct = n_correct * 1.0 / (n_total * 1.0)

    ce = -1.0 * np.dot(targets.T, np.log(1 - y)) - 1.0 * np.dot(1 - targets.T, np.log(y))
    # TODO: Finish this function
    return ce[0,0], frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """
    # TODO: Finish this function
    n_data = len(data)
    dim_data = len(data[0])

    f = 0
    y = logistic_predict(weights, data)

    data = mod_data(data)

    # dl/dw_j = SUM(x_ij * (t_i - (1 - sigmoid(z))))
    df = np.dot(data.T, (1.0 * targets) - (1 - y))

    # to calculate f, we need to sum the negative log of all y iff target is 0 and (1-y) iff target is 1
    f = -1.0 * np.dot(targets.T, np.log(1 - y)) - 1.0 * np.dot(1 - targets.T, np.log(y))

    # calculate P(C=0|x_i) for all x_i 
    return f[0,0], df, y

def get_ndarray(l):
    a = np.ndarray(shape=(len(l), 1))
    for i in range(len(a)):
        a[i] = l[i]
    return a

def get_z_array(weights, data):
    return np.dot(data, weights)

def get_z(weights, data):
    d = data + [1]
    p = 0
    for i in range(len(d)):
        p += weights[i][0] * d[i]
    return p

def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # get regularizer and original logistic return values
    regularizer = hyperparameters['weight_regularization']

    E, df, y = logistic(weights, data, targets, hyperparameters)

    # sum of all weights squared multiplied by lambda/2. Add on top of logistic
    pen_1 = regularizer * 0.5 * (reduce(lambda x,y: x + y * y, weights))
    f = E + pen_1

    # calculat pen for dL/dwi - dE/dwi, add the difference to df
    df = df + regularizer * weights

    return f, df, y
