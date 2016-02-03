#! /usr/bin/env python

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

import sys

if len(sys.argv) != 3:
    print 'Usage: python accuracy_experiment.py <# bins> <# trials>'
    sys.exit(1)

NUM_BINS = int(sys.argv[1])
NUM_TRIALS = int(sys.argv[2])

NUM_TRAINING_EXAMPLES = 100

def single_run(num_bins):
    t_truth = np.random.uniform()
    print 't_truth: ', t_truth
    single_features = np.random.uniform(0, 1, NUM_TRAINING_EXAMPLES)
    #print single_features
    #X_train = [[x] for x in single_features]
    #print X_train
    bins = np.linspace(0, 1, NUM_BINS + 1)
    #print bins
    X_train_discretized = [[bins[i]] for i in np.digitize(single_features, bins)]
    #print X_train_discretized

    y_train = [1.0 if single_features[i] - t_truth > 0 else 0.0 for i in
               range(NUM_TRAINING_EXAMPLES)]
    #print y_train

    #regr = DecisionTreeClassifier(max_depth=1)
    #regr.fit(X_train, y_train)
    #y_train_pred = regr.predict(X_train)
    #train_accuracy = accuracy_score(y_train, y_train_pred)
    #print 'Train Accuracy: ', train_accuracy

    regr_discretized = DecisionTreeClassifier(max_depth=1)
    regr_discretized.fit(X_train_discretized, y_train)
    y_train_pred_discretized = regr_discretized.predict(X_train_discretized)
    #print y_train_pred_discretized
    train_accuracy = accuracy_score(y_train, y_train_pred_discretized)
    print 'Train Accuracy Discretized: ', train_accuracy
    return train_accuracy

print 'Training discretized decision stump for %d trials' % (NUM_TRIALS)
accuracies = [single_run(NUM_BINS) for i in xrange(NUM_TRIALS)]
print 'Accuracy Mean for %d bins: %f' % (NUM_BINS, np.mean(accuracies))

