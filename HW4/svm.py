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
from sklearn.svm import SVC
from baseClassifier import baseClassifier

class SVM(baseClassifier):
    ''' Decision tree classifier  wrapper '''

    def __init__(self, kernel_num=0):
        ''' Init classifier '''
        kernel = ['linear', 'rbf', 'poly', 'sigmoid'][kernel_num]
        self.classifier = SVC(kernel=kernel, max_iter=10000000)

    def fit(self, dataset):
        '''  '''
        return self.classifier.fit(dataset.features, dataset.target)

    def predict(self, features):
        ''''''
        return self.classifier.predict(features)