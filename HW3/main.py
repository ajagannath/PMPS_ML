"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import sys
import time
import math
import struct
import numpy as np
import NNClassifier as nn

def read_idx(filename):
    ''' Method to read data from file '''
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))

        # There is an error in the training file
        # data_type is supposed to give the correct "0x0D" for float64, but its always 8.
        # Hence this workaround
        if dims > 1:
            data_type = 'float64'
        else:
            data_type = 'byte'

        return np.fromstring(f.read(), dtype=np.dtype(data_type).newbyteorder('>')).reshape(shape)

def normalize_input(image_data):
    ''' Input normalization '''
    std = np.std(image_data)
    mean = np.mean(image_data)
    ret_val = (image_data - mean) / std
    return ret_val

def organize_data(image_components_file, labels_file):
    ''' Read the file into usable data structure '''

    image_data = read_idx(image_components_file)
    image_data = normalize_input(image_data)
    label_data = read_idx(labels_file)

    data = []
    for i in range(len(label_data)):
        target = [0.0 for i in range(10)]
        target[label_data[i]] = 1.0
        data.append({"input": np.array([image_data[i]]), "target": np.array([target])})
    
    return data


def main(train_images, train_labels, test_images, test_labels):
    """ Main function """

    # Load the train set
    start_time = time.time()
  #  print("Loading Train data: %s and %s" % (train_images, train_labels))
    train_data = organize_data(train_images, train_labels)
    time_at_train_load = time.time()
  #  print("Training Loaded: %.3f secs!" % (time_at_train_load - start_time))
  
    # Load the test set
  #  print("Loading Test data: %s and %s" % (test_images, test_labels))
    test_data = organize_data(test_images, test_labels)
    time_at_test_load = time.time()
  #  print("Test Loaded: %.3f secs!" % (time_at_test_load - time_at_train_load))




    # Train
    # Test data is used as validation here. ( Don't do that in real life ) 
    network = nn.train_network(train_data, test_data)

    # Final Test
    train_error = nn.test_network(network, train_data)
    test_error = nn.test_network(network, test_data)

    print("Final train MSE: %.3f" % (train_error[0]))
    print("Final train CE: %.3f"  % (train_error[1]))

    print("Final test MSE: %.3f" % (test_error[0]))
    print("Final test CE: %.3f" % (test_error[1]))

    end_time = time.time()
    print("Total execution time: %.2f secs" % (end_time - start_time))

    nn.plot_errors()

if __name__ == "__main__":
    ''' Start the program here '''

    if len(sys.argv) != 5:
        print("Please provide Training and Test files to work with!")
        print("Usage: $python3 main.py train-images-pca.idx2-double train-labels.idx1-ubyte t10k-images-pca.idx2-double t10k-labels.idx1-ubyte")
        exit()

    # Parse arguments 
    train_images = sys.argv[1]
    train_labels = sys.argv[2]
    test_images = sys.argv[3]
    test_labels = sys.argv[4]
    
    # Run the program!
    main(train_images, train_labels, test_images, test_labels)
