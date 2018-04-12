"""
Main function module
"""
#!/usr/bin/env python3

import sys
import math

def combination(K,n):
    ''' K choose n'''
    K_fact = math.factorial(K)
    n_fact = math.factorial(n)
    K_minus_n_fact = math.factorial(K-n)
    ret = K_fact / (n_fact * K_minus_n_fact)
    return ret

def pdf(K, n, p):
    ''''
        Probability that, 
        EXACTLY n outcomes of K are true. with probablity of n = p
    '''
    K_choose_n = combination(K,n)
    p_pow_n = math.pow(p, n)
    q_pow_K_minus_n = math.pow((1-p), (K-n))
    ret = K_choose_n * p_pow_n * q_pow_K_minus_n
    return ret

def cdf(K, n, p):
    ''''
        Probability that, 
        ATLEAST n outcomes of K are true. with probablity of n = p
    '''

    sum = 0.0
    for i in range(n) :
       # print(pdf(K, (n+i), p))
        if n+i > K:
            break
        sum = sum + pdf(K, (n+i), p)
    
    return sum



if __name__ == "__main__":
    ''' Start the program here '''

    # Parse arguments 
    K = int(sys.argv[1])
    n = int(sys.argv[2])
    p = float(sys.argv[3])

    print(cdf(K,n,p))
