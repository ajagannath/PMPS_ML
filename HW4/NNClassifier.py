"""
Neral net module
"""
#!/usr/bin/env python3

import multiprocessing
import math
import copy
import operator as op
import numpy as np
from baseClassifier import baseClassifier

class NeuralNetwork:
    ''' Class to hold complete network '''

    def get_random_matrix(self, m, n, r):
        ''' generate random mattrix mxn with each values in range '''
        mat = []
        for i in range(m):
            #mat.append(np.random.uniform(r[0],r[1],n))
            mat.append([np.random.uniform(r[0],r[1]) for j in range(n)])
        return np.array(mat)

    def init_weights(self, scheme, shape, bounds):
        '''
            Weights initilization
            scheme: 'random'   -> random numbers
                    'uniform'  -> follows uniform distribution
                    'gaussian' -> follows gaussian distribution
            shape : tuple (m,n) -> gets m x n output
            bounds: tuple (r1, r2)
                        range bounds in case of random and uniform
                        mean and std in case of gaussian
        '''

        r1 = bounds[0]
        r2 = bounds[1]

        if scheme == 'gaussian':
            ret_val = np.random.normal(r1, r2, shape)
        elif scheme == 'uniform':
            ret_val = np.random.uniform(r1, r2, shape)
        else:
            ret_val = r1 * np.random.random_sample(shape) + r2

        return ret_val


    def __init__(self, numInput, numHidden, numOutput, wInitScheme, wInitBounds, eeta, alpha, useReLU):
        '''
            Initialization method
            numInput    : number of input units
            numHidden   : number of hidden units
            numOutput   : number of output units
            wInitScheme : weights initialization scheme ( random, uniform, gaussian)
            wInitBounds : bounds for initial weights
            eeta        : initial learning rate
            alpha       : initial momentum term
            useReLu     : weather to use ReLU or sigmoid
        '''

        # Supress numpy warnings
        np.seterr('ignore')

        # Tuning params
        self.eeta = eeta
        self.alpha = alpha
        self.useReLU = useReLU

        # Weights matrix shape ( +1 is for bias)
        wHiddenShape = ((numInput+1), numHidden)
        wOutputShape = ((numHidden+1), numOutput)

        # Initialize weight matricies
        self.weightsHidden = self.init_weights(wInitScheme, wHiddenShape, wInitBounds )
        self.weightsOutput = self.init_weights(wInitScheme, wOutputShape, wInitBounds )

        # Initlailize weight update matricies
        self.updateHidden = np.zeros(wHiddenShape)
        self.updateOutput = np.zeros(wOutputShape)

        # Outputs before sigmoid
        self.actHidden = np.zeros((1, numHidden))
        self.actOutput = np.zeros((1, numOutput))

        # Outputs after sigmoid
        self.outHidden = np.zeros((1, numHidden+1))
        self.Output = np.zeros((1, numOutput))

        # All 1s for input, later we will copy input from the second position
        self.Input = np.ones((1, numInput+1))

    def sigmoid(self, arr):
        ''' caluclate sigmoid '''
        return 1.0 / (1.0 + np.exp(-arr))

    def relu(self, arr):
        ''' Do the ReLu = max(0, x) '''
        ret = np.clip(arr, 0.0, math.inf)
        return ret

    def forward_feed(self, input):
        ''' Takes in input (1x50) and computes the output '''

        # The way this fits into matrix multiplications is like this:
        #
        #                        units --->      51 x 10
        #                       weigths |     <weight matrix>
        #                               | W0 | 0 1 2 .....9 |
        #             1 x 51            | W1 | 1            |       1 x 10
        #     < Inputs with x0 = 1>     v    | 2            |     < Outputs>
        #      [0 1 2 3 ... 50]    x         | 3            |  =  [0 1 .. 9]
        #                                    | .            |
        #                                    | .            |
        #                                    | .            |
        #                                 Wn | 50           |
        #
        #
        #   Input x weightsHidden  = actHidden ===(sigmoid)===> outHidden x weightsOut = FinalOutput
        #    1x51 .   51x10        =  1x10   ====>( + bias)=====>   1x11  .   11x10    =   1x10

        # input[0] is already 1.0
        self.Input[:, 1:] = input
        self.Input[0,0] = 1.0

        # Summation as dot product
        #    1 x 10                  1 x 51       51 x 10
        self.actHidden = np.dot(self.Input, self.weightsHidden)

        # Use ReLU or sigmoid
        if self.useReLU:
            self.outHidden[:,1:] = self.relu(self.actHidden)
        else:
            self.outHidden[:,1:] = self.sigmoid(self.actHidden)

        # reset the first (bias) as 1.0
        self.outHidden[0,0] = 1.0

        # Final output
        #   1 x 10                    1 x 11           11 x 10
        self.actOutput = np.dot(self.outHidden, self.weightsOutput)

         # Use ReLU or sigmoid
        if self.useReLU:
            self.Output = self.relu(self.actOutput)
        else:
            self.Output = self.sigmoid(self.actOutput)

        return self.Output

    def relu_derivative(self, arr):
        ''' derivative of relu '''
        ret = copy.deepcopy(arr)
        ret[ret  > 0.0] = 1.0
        ret[ret == 0.0] = math.pow(math.e, -20)
        ret[ret  < 0.0] = 0.0
        return ret

    def get_delta_hidden(self, err):
        ''' Calculate the delta for hidden layer weights '''

        #  1 x 11           1x10     11x10 transpose-> 10x11
        summation = np.dot( err, self.weightsOutput.transpose())

        if self.useReLU:
            deltaHidden = self.relu_derivative(self.outHidden[:,1:]) * summation[:,1:]
        else:
             #  1 x 10            1 x 10               1 x 10                     1 x 10
            deltaHidden = self.outHidden[:,1:] * ( 1 - self.outHidden[:,1:] ) * summation[:,1:]

        return deltaHidden


    def get_delta_output(self, err):

        if self.useReLU:
            deltaOutput = self.relu_derivative(self.Output) * err
        else:
            deltaOutput = self.Output * ( 1 - self.Output ) * err

        return deltaOutput

    def back_propagation(self, target, eeta=None, alpha=None):
        '''
            Does error back propagation
            target -> vector : expected output
            eeta -> learning rate
            alpha -> momentum term
        '''

        if not eeta:
            eeta = self.eeta

        if not alpha:
            alpha = self.alpha

        # Error ( numpy arrays are wonderful, you can just do this! )
        err = target - self.Output

        # delta at output
        deltaOutput = self.get_delta_output(err)

        # delta at hidden layer
        deltaHidden = self.get_delta_hidden(deltaOutput)

        # Calculate the updates with momentum
        #   51 x 10                         1 x 51 -> 51 x 1      1 x 10
        self.updateHidden = eeta * np.dot(self.Input.transpose(), deltaHidden) +  alpha * self.updateHidden

        #   10 x 11                            1 x 10 ->  10 x 1      1 x 11
        self.updateOutput = eeta * np.dot(self.outHidden.transpose(), deltaOutput)  +  alpha * self.updateOutput


    def update_weights(self, batch_size, updateHidden=None, updateOutput=None):
        '''
            Updates the weights after a batch has done
            batch_size   -> number of accumulated upates to average it by
            updateHidden -> update for hidden weigths
            updateOutput -> update for output weights
        '''

        # For multi threading, this should already be taken care
        self.updateHidden = self.updateHidden / batch_size
        self.updateOutput = self.updateOutput / batch_size

        if not updateHidden:
            updateHidden = self.updateHidden

        if not updateOutput:
            updateOutput = self.updateOutput

        # Update weights
        #     51 x 10              51 x 10         51 x 10
        self.weightsHidden = self.weightsHidden + updateHidden

        #     10 x 11              10 x 11         10 x 11
        self.weightsOutput = self.weightsOutput + updateOutput

class NNClassifier(baseClassifier):
        
    def __init__(self, nInUnits, nOutputUnits, nHiddenUnits=1000):
        ''' Initialization method '''

        # sigmoid vs ReLU
        useReLU = False

        # Weight init scheme
        wInitScheme = 'random'  # uniform , random or gaussian
        wInitBounds = (0.5, 1.5)

        # Tuning params
        self.eeta  = 0.1
        self.alpha = 0.3

        # Max epochs
        self.max_epochs = 500

        # Batch size and length
        self.batch_size = 100

        # Create a network
        self.network = NeuralNetwork(nInUnits, nHiddenUnits, nOutputUnits, \
                                    wInitScheme, wInitBounds, self.eeta, self.alpha, useReLU)

        # Learning scheme
        self.learning_epoch = 1   #  Number of epochs after which we consider changing things ( None = improvement scheme not used )
        self.best_network = copy.deepcopy(self.network) # final network which has less error
        self.improvement_factor = 10  # eeta = eeta +/- (eeta / factor)
        self.min_error_epoch = 0     # Epoch for which we got lowest error    
        self.max_epoch_distance = 20 # number of epochs after which we stop if there is no improvement
        self.last_error = None
        self.last_best_error = None
        self.improved_count = 0
        self.stop_count_down = 10

        # Store errors to plot
        self.train_errors_ce = []
        self.train_errors_mse = []
        self.test_errors_ce = []
        self.test_errors_mse = []
        self.half_epochs = []

    def run_batch(self, network, data_batch, eeta, alpha):
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
        
    def fit(self, train_data, validation_data):
        ''' Train the network in batch '''

        self.train_length = len(train_data)
        self.train_data = train_data
        self.validation_data = validation_data

        # All the initial values
        network        = self.network
        max_epochs     = self.max_epochs
        train_length   = self.train_length
        batch_size     = self.batch_size
        eeta           = self.eeta
        alpha          = self.alpha
        learning_epoch = self.learning_epoch

        for epoch in range(max_epochs):
            num_batches = math.ceil(train_length / batch_size)
            for batch_num in range(num_batches):
                # split the training data
                data_batch = train_data[ batch_num * train_length // num_batches : (batch_num+1) * train_length // num_batches ]
                network = self.run_batch(network, data_batch, eeta, alpha)

                # is it half epoch point?
                if num_batches / (batch_num+1) == 2:
                    self.calculate_errors(network, (epoch+0.5))
            
            # After full epoch
            validation_error = self.calculate_errors(network, (epoch+1))

            # Store the initial error
            if epoch == 0:
                self.last_error = validation_error
                self.last_best_error = validation_error
            
            # Do we want to change how we learn in this epoch?
            if learning_epoch and ((epoch+1) % learning_epoch == 0):
                    batch_size, eeta, alpha = self.improve_learning(validation_error, batch_size, eeta, alpha)
            
            # Determine when to stop and when stopping, get the best performing network
            network, stop = self.determine_stop(network, validation_error, epoch)
            
            if stop:
                break

        return network

    def determine_stop(self, network, current_error, epoch):
        ''' Determne when to stop '''
        
        last_error = self.last_error
        stop = False

        if current_error[1] < self.last_best_error[1]:
            # Network is doing good, learn faster
            self.best_network = copy.deepcopy(network)
            self.last_best_error = current_error
            self.min_error_epoch = epoch
            self.stop_count_down = self.stop_count_down + 1
        elif current_error[1] < last_error[1]:
            self.stop_count_down = self.stop_count_down + 1
        elif current_error[1] == self.last_error[1]:
            pass
        else:
            self.stop_count_down = self.stop_count_down - 1

        self.last_error = current_error

        #print("%d:%.2f" % (epoch, current_error[1]), end='\t', flush=True)

        isLastEpoch = ((epoch+1) == self.max_epochs)
        epoch_distance = epoch - self.min_error_epoch
        
        if self.stop_count_down == 0 or isLastEpoch or epoch_distance > self.max_epoch_distance :
            print("\nStopping, Total Epochs: ", epoch+1)
            network = self.best_network
            stop = True
        
        # Cap count down to 10
        if self.stop_count_down > 10:
            self.stop_count_down = 10
        
        return (network, stop)

    def improve_learning(self, current_error, batch_size, eeta, alpha):
        ''' Implementation of learning scheme '''

        if current_error[1] < self.last_best_error[1]:
            # Network is doing good, learn faster
            self.improved_count = self.improved_count + 1
            eeta = eeta + ( eeta / self.improvement_factor )
        elif current_error[1] < self.last_error[1]:
            # Not soo good, still can learn a bit faster
            self.improved_count = self.improved_count + 1
            eeta = eeta + ( eeta / (self.improvement_factor * 10) )
        elif current_error[1] == self.last_error[1]:
            pass
        else:
            # reduce learning rate and batch size
            self.improved_count = self.improved_count - 1
            eeta = eeta - ( eeta / (self.improvement_factor * 10) )
            batch_size = self.get_smaller_batch_size(batch_size)
            
        # print("At improve",self.improved_count, batch_size, self.last_best_error)
        
        return (batch_size, eeta, alpha)

    def get_smaller_batch_size(self, current_batch_size):
        ''' Next less batch size '''

        if current_batch_size < 2:
            return current_batch_size

        mid_epoch_length = self.train_length // 2 
        batch_size = current_batch_size - 1
        
        while mid_epoch_length % batch_size != 0:
            batch_size = batch_size - 1
            if batch_size < 2:
                break

        return batch_size

    def calculate_errors(self, network, epoch):
        ''' 
            Plotting error every half epoch 
            returns validation error
        '''
        
        train_data = self.train_data
        validation_data = self.validation_data

        train_error = self.test_network(network, train_data)
        test_error = self.test_network(network, validation_data)

        self.train_errors_mse.append(train_error[0])
        self.train_errors_ce.append(train_error[1])
        
        self.test_errors_mse.append(test_error[0])
        self.test_errors_ce.append(test_error[1])

        self.half_epochs.append(epoch)

        return test_error

    def test_network(self, network, data):
        ''' Method to test all samples '''

        mse_arr = np.zeros((1,2))
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

    def predict(self, test_features):
    
        output = []
        for record in test_features:
            output_arr = self.best_network.forward_feed(record)
            out = np.argmax(output_arr)
            output.append(str(out))
        
        return output