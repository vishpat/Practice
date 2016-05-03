#!/usr/bin/env python

"""
Solution to Euler's problem # 3
"""

if __name__ == "__main__":
    num = 600851475143
    factors = list()
    i = 2
    while i < num:
        if num % i == 0:
            factors.append(i)
            num = num / i
            i = 2
        i += 1
    factors.append(i)
    print factors
