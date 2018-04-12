"""
Bias and Variance for a classifier
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import sys
import time
import math
import copy
from dtree import *
from bagging import *
from dataset import *

def biasVariance(classifier, trainset, testset, nBootStraps=100):
    ''''''

    # lets use the inited classifier given to create multiple copies
    classifiers = [copy.deepcopy(classifier) for i in range(nBootStraps)]

    # Arrays to hold training and test predictions of each classifier
   # train_predictions = []
    test_predictions = []

    for c in classifiers:
        # create bootstrap samples and train each classifier
        sampling = trainset.generate_sampling()
        c.fit(sampling)

        # Get predictions for data it is trained on
       # trPreds = c.predict(sampling.features)
       # train_predictions.append(trPreds)

        # Get predictions for testdata
        tePreds = c.predict(testset.features)
        test_predictions.append(tePreds)

    # change matrix to list by sample results
  #  train_predictions = np.array(train_predictions).transpose()
    test_predictions = np.array(test_predictions).transpose()

    # loss, bias, variance
    loss = 0.0
    bias = 0.0
    varp = 0.0
    varn = 0.0
    varc = 0.0

    # For each example in test set
    for record in range(testset.num_records):
        actual_result = testset.target[record]
        predicted_results = test_predictions[record]

        # --------------- #
        # biasvarx function bellow
        one_count = np.count_nonzero(predicted_results == '1')
        zero_count = np.count_nonzero(predicted_results == '0')
        if zero_count > one_count:
            majClass = '0'
            nmax = zero_count
        else:
            majClass = '1'
            nmax = one_count

        if actual_result == '1':
            numCorrectPredictions = one_count
        else:
            numCorrectPredictions = zero_count

        lossx = 1.0 - numCorrectPredictions / nBootStraps
        biasx = float(majClass != actual_result)
        varx = 1.0 - nmax / nBootStraps
        # ---------------- #

        loss = loss + lossx
        bias = bias + biasx

        if biasx != 0.0:
            varn = varn + varx
            varc = varc + 1.0
            varc = varc - lossx
        else:
            varp = varp + varx

    loss = loss / testset.num_records
    bias = bias / testset.num_records
    var = loss - bias
    varp = varp / testset.num_records
    varn = varn / testset.num_records
    varc = varc / testset.num_records

    return {'loss': loss, 'bias': bias, 'var': var, 'varp': varp, 'varn': varn, 'varc': varc}
