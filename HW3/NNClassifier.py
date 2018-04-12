"""
Neral net module
"""
#!/usr/bin/env python3

import multiprocessing
import math
import copy
import operator as op
import numpy as np
import matplotlib.pyplot as plt
import NeuralNet as NN

class NNShared:
    pass
__m = NNShared()

def init(train_data, test_data):
    ''' Initialization method '''

    # Get training and test(validation) data
    __m.train_data = train_data
    __m.test_data = test_data

    # sigmoid vs ReLU
    useReLU = False

    # Num of perceptrons at each level
    nInUnits     = 50
    nHiddenUnits = 10
    nOutputUnits = 10

    # Weight init scheme
    wInitScheme = 'uninform'  # uniform , random or gaussian
    
    if wInitScheme == 'gaussian':
        # Making starting weights similar to input
        train_samples = np.array([record['input'] for record in train_data])
        mu  = np.mean(train_samples)
        std = np.std(train_samples)
        wInitBounds = (mu, std)
    else:
        wInitBounds = (-0.5, 0.5)

    # Tuning params
    __m.eeta  = 0.5
    __m.alpha = 0.5

    # Max epochs
    __m.max_epochs = 500

    # Batch size and length
    __m.train_length = len(train_data)
    __m.batch_size = __m.train_length

    # Create a network
    __m.network = NN.NeuralNetwork(nInUnits, nHiddenUnits, nOutputUnits, \
                                  wInitScheme, wInitBounds, __m.eeta, __m.alpha, useReLU)

     # Learning scheme
    __m.learning_epoch = 1   #  Number of epochs after which we consider changing things ( None = improvement scheme not used )
    __m.best_network = copy.deepcopy(__m.network) # final network which has less error
    __m.improvement_factor = 10  # eeta = eeta +/- (eeta / factor)
    __m.min_error_epoch = 0     # Epoch for which we got lowest error    
    __m.max_epoch_distance = 20 # number of epochs after which we stop if there is no improvement
    __m.last_error = None
    __m.last_best_error = None
    __m.improved_count = 0
    __m.stop_count_down = 5

    # Store errors to plot
    __m.train_errors_ce = []
    __m.train_errors_mse = []
    __m.test_errors_ce = []
    __m.test_errors_mse = []
    __m.half_epochs = []

def run_batch(network, data_batch, eeta, alpha):
    ''' Run the batch and update the network '''

    # Run the forward feed and back propagation
    # for all the record in the batch
    for record in data_batch:
        input  = record['input']
        target = record['target']
        network.forward_feed(input) 
        network.back_propagation(target, eeta, alpha)

    # Update the network weights
    network.update_weights(len(data_batch))

    return network
    
def train_network(train_data, test_data):
    ''' Train the network in batch '''

    # init network
    init(train_data, test_data)

    # Shuffle train data
    # np.random.shuffle(__m.train_data)

    # All the initial values
    network        = __m.network
    max_epochs     = __m.max_epochs
    train_length   = __m.train_length
    batch_size     = __m.batch_size
    eeta           = __m.eeta
    alpha          = __m.alpha
    learning_epoch = __m.learning_epoch

    for epoch in range(max_epochs):
        num_batches = math.ceil(train_length / batch_size)
        for batch_num in range(num_batches):
            # split the training data
            data_batch = train_data[ batch_num * train_length // num_batches : (batch_num+1) * train_length // num_batches ]
            network = run_batch(network, data_batch, eeta, alpha)

            # is it half epoch point?
            if num_batches / (batch_num+1) == 2:
                calculate_errors(network, (epoch+0.5))
        
        # After full epoch
        validation_error = calculate_errors(network, (epoch+1))

        # Store the initial error
        if epoch == 0:
            __m.last_error = validation_error
            __m.last_best_error = validation_error
        
        # Do we want to change how we learn in this epoch?
        if learning_epoch and ((epoch+1) % learning_epoch == 0):
                batch_size, eeta, alpha = improve_learning(validation_error, batch_size, eeta, alpha)
        
        # Determine when to stop and when stopping, get the best performing network
        network, stop = determine_stop(network, validation_error, epoch)
        
        if stop:
           break

    return network

def determine_stop(network, current_error, epoch):
    ''' Determne when to stop '''
    
    last_error = __m.last_error
    stop = False

    if current_error[1] < __m.last_best_error[1]:
        # Network is doing good, learn faster
        __m.best_network = copy.deepcopy(network)
        __m.last_best_error = current_error
        __m.min_error_epoch = epoch
        __m.stop_count_down = __m.stop_count_down + 1
    elif current_error[1] < last_error[1]:
        __m.stop_count_down = __m.stop_count_down + 1
    elif current_error[1] == __m.last_error[1]:
        pass
    else:
        __m.stop_count_down = __m.stop_count_down - 1

    __m.last_error = current_error

    print("%d:%.2f" % (epoch, current_error[1]), end='\t', flush=True)

    isLastEpoch = ((epoch+1) == __m.max_epochs)
    epoch_distance = epoch - __m.min_error_epoch
    
    if __m.stop_count_down == 0 or isLastEpoch or epoch_distance > __m.max_epoch_distance :
        print("\nStopping, Total Epochs: ", epoch+1)
        network = __m.best_network
        stop = True
    
    # Cap count down to 10
    if __m.stop_count_down > 10:
        __m.stop_count_down = 10
    
    return (network, stop)

def improve_learning(current_error, batch_size, eeta, alpha):
    ''' Implementation of learning scheme '''

    if current_error[1] < __m.last_best_error[1]:
        # Network is doing good, learn faster
        __m.improved_count = __m.improved_count + 1
        eeta = eeta + ( eeta / __m.improvement_factor )
    elif current_error[1] < __m.last_error[1]:
        # Not soo good, still can learn a bit faster
        __m.improved_count = __m.improved_count + 1
        eeta = eeta + ( eeta / (__m.improvement_factor * 10) )
    elif current_error[1] == __m.last_error[1]:
        pass
    else:
        # reduce learning rate and batch size
        __m.improved_count = __m.improved_count - 1
        eeta = eeta - ( eeta / (__m.improvement_factor * 10) )
        batch_size = get_smaller_batch_size(batch_size)
        
   # print("At improve",__m.improved_count, batch_size, __m.last_best_error)
     
    return (batch_size, eeta, alpha)

def get_smaller_batch_size(current_batch_size):
    ''' Next less batch size '''

    if current_batch_size < 2:
        return current_batch_size

    mid_epoch_length = __m.train_length // 2 
    batch_size = current_batch_size - 1
    
    while mid_epoch_length % batch_size != 0:
        batch_size = batch_size - 1
        if batch_size < 2:
            break

    return batch_size

def calculate_errors(network, epoch):
    ''' 
        Plotting error every half epoch 
        returns validation error
    '''
    
    train_data = __m.train_data
    test_data = __m.test_data

    train_error = test_network(network, train_data)
    test_error = test_network(network, test_data)

    __m.train_errors_mse.append(train_error[0])
    __m.train_errors_ce.append(train_error[1])
    
    __m.test_errors_mse.append(test_error[0])
    __m.test_errors_ce.append(test_error[1])

    __m.half_epochs.append(epoch)

    return test_error

def test_network(network, data):
    ''' Method to test all samples '''

    mse_arr = np.zeros((1,10))
    ce = 0.0
    for record in data:
        input  = record['input']
        target_arr = record['target']
        output_arr = network.forward_feed(input)

        err = np.abs(target_arr - output_arr)
        mse_arr = mse_arr + (err * err)

        target = np.argmax(target_arr)
        output =  np.argmax(output_arr)
        if target != output:
            ce = ce + 1


    mse = mse_arr.sum() / len(data)
    ce = 100.0 * ce / len(data)

    return (mse, ce)

def plot_errors():
    ''' Plot errors that we collected every half epoch '''


    print(__m.train_errors_mse)
    print(__m.train_errors_ce)
    print('-------')
    print(__m.test_errors_mse)
    print(__m.test_errors_ce)
    print('-------')
    print(__m.half_epochs)


    # Train data plots
    train_mse_figure = plt.figure()
    train_ce_figure = plt.figure()
    train_mse_plot = train_mse_figure.add_subplot(111)
    train_ce_plot  = train_ce_figure.add_subplot(111)

    train_mse_plot.scatter(__m.half_epochs, __m.train_errors_mse)
    train_mse_plot.set_xlabel('Epochs')
    train_mse_plot.set_ylabel('Train MSE')
   
    train_ce_plot.scatter(__m.half_epochs, __m.train_errors_ce)
    train_ce_plot.set_xlabel('Epochs')
    train_ce_plot.set_ylabel('Train 1/0')

    # Test data plots
    test_mse_figure = plt.figure()
    test_ce_figure = plt.figure()
    test_mse_plot = test_mse_figure.add_subplot(111)
    test_ce_plot  = test_ce_figure.add_subplot(111)
    
    test_mse_plot.scatter(__m.half_epochs, __m.test_errors_mse)
    test_mse_plot.set_xlabel('Epochs')
    test_mse_plot.set_ylabel('Test MSE')

    test_ce_plot.scatter(__m.half_epochs, __m.test_errors_ce)
    test_ce_plot.set_xlabel('Epochs')
    test_ce_plot.set_ylabel('Test 1/0')

    plt.show()
