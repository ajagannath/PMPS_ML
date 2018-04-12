"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import sys
import time
import math

import random
import numpy as np
from dtree import *
from baseClassifier import baseClassifier

class BagClassifier(baseClassifier):
    ''' Classifier uses different number of ID3 and votes '''

    def __init__(self, num_classifiers, max_depth=None):
        '''  '''
        self.classifiers = []
        for i in range(num_classifiers):
            self.classifiers.append(DTree(max_depth)) 
    
    def fit(self, dataset, resampling=True, sampled_sets=[]):
        ''''''
        
        if len(sampled_sets) == len(self.classifiers):
            for i in range(len(sampled_sets)):
                self.classifiers[i].fit(sampled_sets[i], False)
        else:
            for classifier in self.classifiers:
                classifier.fit(dataset, resampling)

    def predict(self, features):
        ''' Aggregate the results from all classifiers and vote for the classification '''

        classifiers_result = []
        for classifier in self.classifiers:
            classifiers_result.append(classifier.predict(features))

        # max vote for each prediction
        result_by_samples = np.array(classifiers_result).transpose()
        predictions = []
        for result in result_by_samples:
            one_count = np.count_nonzero(result == '1')
            zero_count = np.count_nonzero(result == '0')

            if one_count > zero_count:
                predictions.append('1')
            elif one_count < zero_count:
                predictions.append('0')
            else:
                flip = random.randint(0,1)
                predictions.append(str(flip))
        
        return predictions