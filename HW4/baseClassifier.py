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

class baseClassifier:
    ''' Base classifier which has common methods for error calculatation etc '''

    def __init__(self):
        pass

    def error(self, predictions, target):
        ''' Calculates accuracy and errors given predictions and target '''

        total_records = len(predictions)
        right_predictions = 0

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(total_records):
            predicted_value = predictions[i]
            actual_value    = target[i]

            if actual_value == predicted_value:
                right_predictions += 1
                if predicted_value == '1':
                    true_positives += 1
            else:
                if predicted_value == '1':
                    false_positives += 1
                else:
                    false_negatives += 1
        
        accuracy  = 100 * right_predictions / total_records
       # precision = 100 * true_positives / (true_positives + false_positives)
        #recall    = 100 * true_positives / (true_positives + false_negatives)

        return {"accuracy" : accuracy} # "precision" : precision, "recall": recall}
