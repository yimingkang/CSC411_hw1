from utils import *
from run_knn import run_knn
from plot_digits import *

def get_perc(test, target):
    total = 0
    correct = 0
    for i in range(len(test)):
        if test[i][0] == target[i][0]:
            correct += 1
        total += 1
    return correct * 1.0/(total * 1.0) * 100


def main():
    train_data, train_labels = load_train()
    test, target = load_valid()

    print "VALIDATION"
    for i in [1, 3, 5, 7, 9]:
        labels = run_knn(i, train_data, train_labels, test)
        #plot_digits(test)
        print "K = ", i, " perc = ", get_perc(labels, target)

    test, target = load_test()
    print "TEST"
    for i in [1, 3, 5, 7, 9]:
        labels = run_knn(i, train_data, train_labels, test)
        #plot_digits(test)
        print "K = ", i, " perc = ", get_perc(labels, target)


if __name__ == '__main__':
    main()


