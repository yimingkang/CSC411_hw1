import numpy as np
import matplotlib.pyplot as pt
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *

class file_printer:
    def __init__(self, file_name, use_plot=True, delimiter=','):
        self.plt = use_plot
        self.f = open(file_name, 'w')
        self.delimiter = delimiter
        self.series = None

    def println(self, line):
        self.f.write(line + '\n')

    def printlist(self, ls):
        if self.plt:
            if self.series is None:
                self.series = [[] for i in ls]
            else:
                for i in range(len(ls)):
                    self.series[i].append(ls[i])
        x = [str(i) for i in ls]
        line = self.delimiter.join(x)
        self.println(line)

    def closef(self):
        self.f.close()

    def show(self, title, log=False):
        pt.title(title)
        pt.legend()
        if log:
            pt.xscale('log')
        pt.show()

    def plot(self, x, y, label):
        if max(x, y) > len(self.series) - 1:
            raise ValueError("Plot index out of bound")
        else:
            if x != -1:
                pt.plot(self.series[x], self.series[y], label=label)
            else:
                pt.plot(range(1, len(self.series[y]) + 1), self.series[y], label=label)

def run_logistic_regression():
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape
    output_file = file_printer("train_output.log")
    output_file.printlist(['pen(lambda)', 'train_ce', 'train_predict_perc', 'validate_ce', 'validate_predict_perc'])

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.1, 
                    'weight_regularization': 0.0001,  
                    'num_iterations': 500,
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.randn(M+1, 1)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    for i in range(1, 6):
        hyperparameters['weight_regularization'] *= 10
        for t in xrange(hyperparameters['num_iterations']):

            # TODO: you may need to modify this loop to create plots, etc.

            # Find the negative log likelihood and its derivatives w.r.t. the weights.
            f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
            
            # Evaluate the prediction.
            cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

            if np.isnan(f) or np.isinf(f):
                raise ValueError("nan/inf error")

            # update parameters
            weights = weights - hyperparameters['learning_rate'] * df / N

            # Make a prediction on the valid_inputs.
            predictions_valid = logistic_predict(weights, valid_inputs)

            # Evaluate the prediction.
            cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
            
            if t == hyperparameters['num_iterations'] - 1:
                # print stats on final iteration
                output_file.printlist([
                            hyperparameters['weight_regularization'],
                            cross_entropy_train,
                            frac_correct_train * 100,
                            cross_entropy_valid,
                            frac_correct_valid * 100,
                ])  
                print "Final stats: ", cross_entropy_train, frac_correct_valid * 100, cross_entropy_valid, frac_correct_valid * 100

    output_file.closef()
    output_file.plot(0, 1, 'CE_train')
    output_file.plot(0, 3, 'CE_validate')
    output_file.show("mnist_train", log=True)

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)

    diff = check_grad(logistic_pen,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    run_logistic_regression()
