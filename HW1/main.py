"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import sys
import time
import json
import arff
from classifier import * 

# There are 2 versions, please use 1 at a time.

#from DTree import * 
from DTree_v2 import *

def main(train_file_path, test_file_path, confidences):
    """ Main function """

    print("Building and testing trees for Confidences: ", confidences)

    if train_file_path == test_file_path:
        print("Train and test files are same!!")
        exit()

    # Load the train set
    start_time = time.time()
    print("Loading Train File:", train_file_path)
    training_file = open(train_file_path)
    training_set = arff.load(training_file)
    time_at_train_load = time.time()
    print("Training Loaded: %.3f secs!" % (time_at_train_load - start_time))
    
    # Load the test set
    print("\nLoading Test File:", test_file_path)
    test_file = open(test_file_path)
    test_set  = arff.load(test_file)
    time_at_test_load = time.time()
    print("Test Loaded: %.3fsecs!" % (time_at_test_load - time_at_train_load))

    # Lets prepare the data to be processed
    training_data = training_set['data']
    training_attribute_details = training_set['attributes']
    training_attribute_indecies = [i for i in range(len(training_set['attributes']))]
   
    # Prepare test data too ( used only in classification later )
    test_data = test_set['data']
    test_attribute_indecies = [i for i in range(len(test_set['attributes']))]
    if training_attribute_indecies[-1] != test_attribute_indecies[-1]:
       print("DEBUG: Target attribute mismatch!\nCheck the data set inputs!")
       exit()

    # Attribute indecies we want to process 
    # Target attribute is the last one!
    non_target_attribute_indecies = training_attribute_indecies[:-1]
    target_attribute_index = training_attribute_indecies[-1]
    print("Target Attribute %d -> %s " % (target_attribute_index, training_attribute_details[target_attribute_index]))
    
    # Construct , analyze  and classify for each confidence level
    for confidence in confidences:
        
        print("\nGrowing Tree for Confidence: %.2f %%" % confidence )
        dtree_start_time = time.time()
        dtree = create_dtree(training_data, training_attribute_details, non_target_attribute_indecies, target_attribute_index, confidence)
        dtree_end_time = time.time()
        print("Tree is ready! (%.2f secs)" % (dtree_end_time - dtree_start_time))

        print("Starting Tree analyze:")
        result = analyze_dtree(dtree)
        analyze_end_time = time.time()
        print("Analysis complete (%.2f secs) " % (analyze_end_time - dtree_end_time))
        print("\tNodes: %d \n\tLeaves: %d \n\tSneakPeak: %s" % (result['num_nodes'], result['num_leaves'], result['node_order']))

        print("Starting Train Classification")
        train_result = classify(training_data, dtree, target_attribute_index)
        train_classification_end_time = time.time()
        print("Training classification complete (%.2f secs)" % (train_classification_end_time - analyze_end_time) )
        print("\tAccuracy: %.2f %% \n\tPrecision: %d %% \n\tRecall: %.2f %%" % (train_result['accuracy'], train_result['precision'], train_result['recall']))

        # Paths taken:
        #print("\tPositive path in Train: %s" % majority_path(train_result['p_paths']))
        #print("\tNegetive path in Train: %s" % majority_path(train_result['n_paths']))

        print("Starting Test classification")
        test_result = classify(test_data, dtree, target_attribute_index)    
        test_classification_end_time = time.time()
        print("Testing classification complete (%.2f secs)" % (test_classification_end_time - train_classification_end_time) ) 
        print("\tAccuracy: %.2f %% \n\tPrecision: %d %% \n\tRecall: %.2f %%" % (test_result['accuracy'], test_result['precision'], test_result['recall']))

        # Paths taken:
        #print("\tPositive path in Test: %s" % majority_path(test_result['p_paths']))
        #print("\tNegetive path in Test: %s" % majority_path(test_result['n_paths']))

    end_time = time.time()
    print("Total execution time: %.2f secs" % (end_time - start_time))

if __name__ == "__main__":
    ''' Start the program here '''

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Please provide Training and Test files to work with!")
        print("Usage: $python3 main.py <path_training_filename.arff> <path_test_filename.arff> [optional confidence_level]")
        print("       if confidence_level is not provided, progem will run for confs: [0, 50, 80, 95, 99] ")
        exit()

    # Parrse arguments 
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    confidence = None
    if len(sys.argv) == 4:
        confidence = float(sys.argv[3])

    if confidence != None:
        confidences = [confidence]
    else:
        # Default confidence levels to test on
        # O confidance takes significantly longer than others so 
        # doing this in reverse!
        # confidences = [99, 95, 80, 50, 0]
        confidences = [81.0, 81.5, 82, 82.5, 83, 83.5, 84, 84.5, 85, 85.5, 86, 86.5]
    
    # Run the program!
    main(train_file_path, test_file_path, confidences)
