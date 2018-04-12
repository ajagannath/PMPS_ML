"""
Neral net module
"""
#!/usr/bin/env python3

import multiprocessing
import math
import operator as op
import numpy as np
import copy
import inputs

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




    # def print_weights(self):
    #     print("HX1")
    #     print (repr(self.weightsHidden[0,:].transpose()))

    #     print("HXI")
    #     print (repr(self.weightsHidden[1:,:].transpose()))


    #     print("OX1")
    #     print (repr(self.weightsOutput[0,:].transpose()))

    #     print("OXH")
    #     print (repr(self.weightsOutput[1:,:].transpose()))
