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

    def show(self, title):
        pt.title(title)
        pt.legend()
        pt.show()

    def plot(self, x, y, label):
        print self.series[y]
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
    output_file.printlist(['train_ce', 'train_predict_perc', 'validate_ce', 'validate_predict_perc'])

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.1, 
                    'weight_regularization': 1,  
                    'num_iterations': 1000,
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.randn(M+1, 1)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    for t in xrange(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        
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
        
        # print stats to file, fuck matplotlib btw
        output_file.printlist([
                    cross_entropy_train,
                    frac_correct_train * 100,
                    cross_entropy_valid,
                    frac_correct_valid * 100,
        ])  

        # print some stats
        stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
        stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
        print stat_msg.format(t+1,
                              float(f / N),
                              float(cross_entropy_train),
                              float(frac_correct_train*100),
                              float(cross_entropy_valid),
                              float(frac_correct_valid*100))
    output_file.closef()
    output_file.plot(-1, 0, "cross_entropy_train")
    output_file.plot(-1, 2, "cross_entropy_validate")
    output_file.show("mnist_train")

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

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    run_logistic_regression()
