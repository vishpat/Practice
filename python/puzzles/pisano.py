#!/usr/bin/env python

import sys

def fibonnaci(n):
    n1 = n2 = fib = 1
    i = 2
    while i < n:
        fib = n1 + n2
        n1_prev = n1
        n1 = n2
        n2 += n1_prev 
        i += 1
    
    return fib 


def fib_mod(n, m):
    fib_n = fibonnaci(n)
    return fib_n % m


def period_verify(period, m):
    for i in range(1, period):
        n1 = fib_mod(i, m)
        n2 = fib_mod(i + period, m)
        if n1 != n2:
            return False

    return True


def pisano(m):
    if m == 1:
        return 1

    period = 2 
    
    while not period_verify(period, m):
        period += 1

    return period        


if __name__ == "__main__":
    print pisano(int(sys.argv[1]))
