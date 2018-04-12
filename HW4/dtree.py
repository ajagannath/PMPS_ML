"""
Decision Tree classifier with sampling
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import sys
import time
import math

import numpy as np
from id3 import Id3Estimator 
from sklearn import tree
from baseClassifier import baseClassifier

class DTree(baseClassifier):
    ''' Decision tree classifier  wrapper '''

    def __init__(self, max_depth=None):
        ''' Init classifier '''
        #self.classifier = Id3Estimator(max_depth=max_depth)
        self.classifier = tree.DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, dataset, resampling=False):
        '''  '''
        if resampling:
            train_data = dataset.generate_sampling()
        else:
            train_data = dataset

        return self.classifier.fit(train_data.features, train_data.target)

    def predict(self, features):
        ''''''
        return self.classifier.predict(features)