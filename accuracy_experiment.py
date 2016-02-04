#! /usr/bin/env python

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from math import log

import sys

if len(sys.argv) != 5:
    print 'Usage: python accuracy_experiment.py <# features> <# bins> <#training examples> <# trials>'
    sys.exit(1)

NUM_FEATURES = int(sys.argv[1])
NUM_BINS = int(sys.argv[2])
NUM_TRAINING_EXAMPLES = int(sys.argv[3])
NUM_TRIALS = int(sys.argv[4])

def single_run(num_bins):
    t_truth = np.random.rand(1, NUM_FEATURES)
    #print 't_truth: ', t_truth
    features = np.random.rand(NUM_TRAINING_EXAMPLES, NUM_FEATURES)
    #print features
    X_train = features
    #print X_train
    bins = np.linspace(0, 1, NUM_BINS + 1)
    #print bins
    X_train_discretized = [[bins[i] for i in np.digitize(row, bins)] for row in features]
    #print X_train_discretized

    if NUM_FEATURES == 1:
        y_train = [1.0 if row[0] else 0.0
                   for row in features > t_truth]
    else:
        y_train = [1.0 if reduce(lambda a, b: np.logical_xor(a, b), row, False)
                       else 0.0
                       for row in features > t_truth]
    #print y_train

    regr = DecisionTreeClassifier(max_depth=NUM_FEATURES)
    regr.fit(X_train, y_train)
    y_train_pred = regr.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print 'Train Accuracy: ', train_accuracy

    regr_discretized = DecisionTreeClassifier(max_depth=NUM_FEATURES)
    regr_discretized.fit(X_train_discretized, y_train)
    y_train_pred_discretized = regr_discretized.predict(X_train_discretized)
    #print y_train_pred_discretized
    train_accuracy_discretized = accuracy_score(y_train, y_train_pred_discretized)
    print 'Train Accuracy Discretized: ', train_accuracy_discretized
    return (train_accuracy, train_accuracy_discretized)

print 'Training discretized decision stump for %d trials' % (NUM_TRIALS)
accuracies, discretized_accuracies = zip(*[single_run(NUM_BINS) for i in xrange(NUM_TRIALS)])
print 'Standard Accuracy Mean: %f' % np.mean(accuracies)
print 'Accuracy Mean for %d bins: %f' % (NUM_BINS, np.mean(discretized_accuracies))

