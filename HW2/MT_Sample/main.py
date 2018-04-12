"""
Main function module
"""
#!/usr/bin/env python3

import time
from random import *
import MTSample as MTS


def main():
    ''' Sample main '''

    data = []

    for i in range(100):
        data.append(randint(1,100))

    row_nums = [i for i in range(100)]

    start_time = time.time()
    MTS.init(data)

    after_init = time.time()
    print("Test_result: ", MTS.test(row_nums))

    after_test = time.time()
    print("Test_MT_result: ", MTS.test_mt(row_nums))

    after_test_mt = time.time()

    print("Test Time:", (after_test - start_time))
    print("Test mT Time:", (after_test_mt - start_time))


if __name__ == "__main__":
    ''' Start the program here '''
    main()