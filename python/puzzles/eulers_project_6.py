#!/usr/bin/env python


def sum_n(n):
    return (n*(n + 1))/2


def sum_nn(n):
    return (n*(n + 1)*(2*n + 1))/6


if __name__ == "__main__":
    square_sum = sum_nn(100)
    sum_100 = sum_n(100)
    print sum_100*sum_100 - square_sum
