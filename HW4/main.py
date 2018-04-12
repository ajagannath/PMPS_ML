"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from dtree import DTree
from dataset import Dataset
from bagging import BagClassifier
from BV import biasVariance
from svm import SVM
from NNClassifier import NNClassifier

def main(train_file_path, test_file_path):
    """ Main function """

    if train_file_path == test_file_path:
        print("Train and test files are same!!")
        exit()

    # Load the train set
    start_time = time.time()
    print("Loading Train File:", train_file_path)
    train_data = Dataset(train_file_path)
    time_at_train_load = time.time()
    print("Training Loaded: %.3f secs!\n" % (time_at_train_load - start_time))
    
    # Load the test set
    print("Loading Test File:", test_file_path)
    test_data  = Dataset(test_file_path) 
    time_at_test_load = time.time()
    print("Test Loaded: %.3fsecs!\n" % (time_at_test_load - time_at_train_load))

    # Run question 1
    #question1_0(train_data, test_data)
    #question1_1(train_data, test_data)
    #question1_2(train_data, test_data)
    
    # Question 4
    #question4_2(train_data, test_data)
    #question4_3(train_data, test_data)
    #question4_4(train_data, test_data)

    question4_4_bonus(train_data, test_data)

   # train_data.get_in_nn_format()
    
    
    end_time = time.time()
    print("Total execution time: %.2f secs" % (end_time - start_time))


def question4_4(train_data, test_data):
    # Question 4.0
    biases = []
    vars = []
    for kernel_num in range(4):
        clf = SVM(kernel_num)
        bv = biasVariance(clf, train_data, test_data, 100)
        biases.append(bv['bias'])
        vars.append(bv['var'])
    print(biases)
    print(vars)

def question4_3(train_data, test_data):
    clf = NNClassifier(len(train_data.features[0]), 2)
    clf.fit(train_data.get_in_nn_format(), test_data.get_in_nn_format())
    preds = clf.predict(test_data.features)
    print(clf.error(preds, test_data.target))

def question4_2(train_data, test_data):
    
    accuracies = []
    for kernel_num in range(4):
        clf = SVM(kernel_num)
        clf.fit(train_data)
        predictions = clf.predict(test_data.features)
        errs = clf.error(predictions, test_data.target)
        accuracies.append(errs['accuracy'])
    print(accuracies)

def question1_0(train_data, test_data):
    # Question 1.0
    test_samplings = [1, 3, 5, 10, 20]
    avg_num = 50    
    avg_acc = []
    for i in range(avg_num):
        accuracies = []
        for test in test_samplings:
            clf = BagClassifier(test)
            clf.fit(train_data, True)
            predictions = clf.predict(test_data.features)
            errs = clf.error(predictions, test_data.target)
            accuracies.append(errs['accuracy'])
        avg_acc.append(accuracies)
    accs = np.mean(np.array(avg_acc), axis=0)
    max_accs = np.max(np.array(avg_acc), axis=0)
    min_accs = np.min(np.array(avg_acc), axis=0)

    plt.plot(test_samplings, accs, c='b')
    plt.scatter(test_samplings, accs, c='b')
    plt.plot(test_samplings, max_accs, c='r')
    plt.scatter(test_samplings, max_accs, c='r')
    plt.plot(test_samplings, min_accs, c='g')
    plt.scatter(test_samplings, min_accs, c='g')
    plt.legend(['Avg accuracy', 'Max Accuracy', 'Min Accuracy'], loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Samplings')
    plt.ylim(ymin=50, ymax=80)
    plt.show()

def question1_1(train_data, test_data):
    # Question 1.1
    # Dtree alone with max depths
    max_depths = [i+1 for i in range(20)]
    biases = []
    vars = []
    for depth in max_depths:
        clf = DTree(depth)
        bv = biasVariance(clf, train_data, test_data, 100)
        biases.append(bv['bias'])
        vars.append(bv['var'])
    
    print(biases)
    print(vars)

    plt.plot(max_depths, biases, c='b')
    plt.scatter(max_depths, biases, c='b')
    plt.plot(max_depths, vars, c='g')
    plt.scatter(max_depths, vars, c='g')
    plt.legend(['Bias', 'Variance'], loc='lower right')
    plt.ylabel('BV')
    plt.xlabel('Max Depth')
    plt.show()

def question1_2(train_data, test_data):
    # Question 1.2
    # Max depth with bagging
    test_samplings = [20] #[1, 3, 5, 10, 20]
    max_depths = [i+1 for i in range(20)]
    
    accuracies = []
    for test in test_samplings:
        samp_acc = []
        biases = []
        vars = []
        for depth in max_depths:
            clf = BagClassifier(test, depth)
            bv = biasVariance(clf, train_data, test_data, 100)
            biases.append(bv['bias'])
            vars.append(bv['var'])

            # avg_num = 50    
            # for i in range(avg_num):
            #     avg_acc = []
            #     clf = BagClassifier(test, depth)
            #     clf.fit(train_data, True)
            #     predictions = clf.predict(test_data.features)
            #     errs = clf.error(predictions, test_data.target)
            #     avg_acc.append(errs['accuracy'])
            
            # samp_acc.append(np.mean(np.array(avg_acc)))
    
        #accuracies.append(samp_acc)
        print("Samplings = ", test)
        print(biases)
        print(vars)
       # print(accuracies)

        plt.plot(max_depths, biases, c='b')
        plt.scatter(max_depths, biases, c='b')
        plt.plot(max_depths, vars, c='g')
        plt.scatter(max_depths, vars, c='g')
        plt.legend(['Bias', 'Variance'], loc='best')
        plt.ylabel('BV for Sampling = 20')
        plt.xlabel('Max Depth')
        plt.show()

    # plt.plot(max_depths, accuracies[0], c='b')
    # plt.scatter(max_depths, accuracies[0], c='b')

    # plt.plot(max_depths, accuracies[1], c='g')
    # plt.scatter(max_depths, accuracies[1], c='g')

    # plt.plot(max_depths, accuracies[2], c='r')
    # plt.scatter(max_depths, accuracies[2], c='r')

    # plt.plot(max_depths, accuracies[3], c='c')
    # plt.scatter(max_depths, accuracies[3], c='c')

    # plt.plot(max_depths, accuracies[4], c='m')
    # plt.scatter(max_depths, accuracies[4], c='m')
    
    # plt.legend(['samplings=1 ', 'samplings=3', 'samplings=5', 'samplings=10', 'samplings=20'], loc='lower right')
    # plt.ylabel('Accuracies')
    # plt.xlabel('Max Depth')
    # plt.ylim(ymin=50, ymax=80)
    # plt.show()

def question4_4_bonus(train_data, test_data):
    # Question 1.1
    # Dtree alone with max depths
    test_samplings = [1, 3, 5, 10, 20]
    biases = []   
    vars = []
    for test in test_samplings:
        clf = BagClassifier(test)
        bv = biasVariance(clf, train_data, test_data, 100)
        biases.append(bv['bias'])
        vars.append(bv['var'])
    
    print(biases)
    print(vars)

if __name__ == "__main__":
    ''' Start the program here '''

    if len(sys.argv) != 3:
        print("Please provide Training and Test files to work with!")
        print("Usage: $python3 main.py pima-indians-diabetes.train.txt pima-indians-diabetes.test.txt")
        exit()

    # Parrse arguments 
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    
    # Run the program!
    main(train_file_path, test_file_path)
