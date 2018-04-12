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

def main(train_file_path):
    """ Main function """

    # Load the train set
    start_time = time.time()
    print("Loading Train File:", train_file_path)
    training_file = open(train_file_path)
    CF.train(csv.reader(training_file))
    time_at_train_load = time.time()
    print("Training Loaded: %.3f secs!" % (time_at_train_load - start_time))

    time_test_start = time.time()
    ratings = CF.predicted_movies('999')


    print("Result: ", ratings)
    time_test_end = time.time()
    print("Testing time %f" % (time_test_end - time_test_start))
   
    end_time = time.time()
    print("Total execution time: %.2f secs" % (end_time - start_time))

if __name__ == "__main__":
    ''' Start the program here '''

    if len(sys.argv) != 2:
        print("Please provide Training and Test files to work with!")
        print("Usage: $python3 main.py TrainingRatings.txt")
        exit()

    # Parrse arguments 
    train_file_path = sys.argv[1]
    
    # Run the program!
    main(train_file_path)
