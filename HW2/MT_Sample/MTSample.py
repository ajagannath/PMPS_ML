"""
Collaborative filterinf module
"""
#!/usr/bin/env python3

import multiprocessing
import math
import os


# Just to store shared data
class Users:
    pass
u = Users()


def init(data):

    print(os.getpid())
    u.data = data
    u.pool = multiprocessing.Pool(processes=4)


def test(row_numbers):
    print(os.getpid())
    sum_n = 0.0
    for row in row_numbers:
        sum_n += u.data[row]

    return sum_n



def test_mt(row_numbers):
    
    num_rows = len(row_numbers)

    results = u.pool.starmap (test, ([row_numbers[:25]], [row_numbers[25:50]], [row_numbers[50:75]],[row_numbers[75:100]]))
    
    sum_n = 0.0
    for result in results:
        sum_n += result

    return sum_n