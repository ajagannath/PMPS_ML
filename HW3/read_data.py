""" A function that can read MNIST's idx file format into numpy arrays.
    The MNIST data files can be downloaded from here:
    
    http://yann.lecun.com/exdb/mnist/
    This relies on the fact that the MNIST dataset consistently uses
    unsigned char types with their data segments.
"""

import struct
import numpy as np
import NuralNet as nn

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))

        # There is an error in the training file
        # data_type is supposed to give the correct "0x0D" fir float64, but its always 8.
        # Hence this workaround
        if dims > 1:
            data_type = 'float64'
        else:
            data_type = 'byte'

        return np.fromstring(f.read(), dtype=np.dtype(data_type).newbyteorder('>')).reshape(shape)