"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import sys
import time
import math
import csv
import collaborativeFiltering as CF

def main(train_file_path, test_file_path):
    """ Main function """

    if train_file_path == test_file_path:
        print("Train and test files are same!!")
        exit()

    # Load the train set
    start_time = time.time()
    print("Loading Train File:", train_file_path)
    training_file = open(train_file_path)
    CF.train(csv.reader(training_file))
    time_at_train_load = time.time()
    print("Training Loaded: %.3f secs!" % (time_at_train_load - start_time))
    
    # Load the test set
    print("\nLoading Test File:", test_file_path)
    test_file = open(test_file_path)
    test_data  = [row for row in csv.reader(test_file)] 
    time_at_test_load = time.time()
    print("Test Loaded: %.3fsecs!" % (time_at_test_load - time_at_train_load))


    time_test_start = time.time()
    mae = 0.0   # Mean absolute error
    rme = 0.0   # Root mean squared error
    #updated_test_data = []

    write_file = "output.csv"
    with open(write_file, "w") as output:
        for row in test_data:
            item = row[0]
            user = row[1]
            vote = float(row[2])
            pvote = CF.predicted_vote_mt(user, item)
            
            update = (item, user, vote, pvote)
            output.write( ','.join([str(x) for x in update]) + '\n')

            err = abs(vote-pvote)
            mae += err
            rme += err * err

    total_tests = len(test_data)
    mae = mae / total_tests
    rme = math.sqrt(rme / total_tests)


    print("Result: MAE: %.3f,  RME: %.3f" %( mae, rme))
    time_test_end = time.time()
    print("Testing time %f" % (time_test_end - time_test_start))
   
    end_time = time.time()
    print("Total execution time: %.2f secs" % (end_time - start_time))

if __name__ == "__main__":
    ''' Start the program here '''

    if len(sys.argv) != 3:
        print("Please provide Training and Test files to work with!")
        print("Usage: $python3 main.py TrainingRatings.txt TestingRatings.txt")
        exit()

    # Parrse arguments 
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    
    # Run the program!
    main(train_file_path, test_file_path)
