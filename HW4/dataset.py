"""
Dataset class
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"


import csv
import math
import random
import numpy as np

class DatasetBase:
    ''' Base for data set '''
    def __init__(self, num_records, features, target):
        ''' Just initing the members '''
        self.num_records = num_records
        self.features = np.array(features)
        self.target = np.array(target)

    def generate_sampling(self):
        ''' Generate a sampling of this dataset '''

        num_records = self.num_records
        sampled_features = []
        sampled_target = []

        for rec_num in range(num_records):
            indx = random.randint(0, num_records-1)
           # indx = math.ceil(random.uniform(0,num_records-1))
            sampled_features.append(self.features[indx])
            sampled_target.append(self.target[indx])

        return DatasetBase(num_records, sampled_features, sampled_target)

    def get_in_nn_format(self):
        num_records = self.num_records
        data_set = []
        for indx in range(num_records):
            target = [0.0 for i in range(2)]
            target[int(self.target[indx])] = 1.0
            data_set.append({'input':np.array([self.features[indx]]), 'target': np.array([target]) })
        
        return data_set

class Dataset(DatasetBase):
    ''' Data set '''
    def __init__(self, csv_file_path):
        ''' Inits dataset by reading the csv file '''

        features = []
        target = []
        csv_file = open(csv_file_path)
        for record in csv.reader(csv_file):
            features.append(record[0:-1])
            target.append(record[-1])
        
        self.num_records = len(features)
        self.features = np.array(features)
        self.target = np.array(target)
    
