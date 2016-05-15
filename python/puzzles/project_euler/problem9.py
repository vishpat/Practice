#!/usr/bin/env python

import math

if __name__ == "__main__":
    k = int(math.sqrt(1000))
    for x in range(1, k):
        for y in range(1, x):
            a = x*x - y*y
            b = 2*x*y
            c = x*x + y*y
            if (c*c == (a*a + b*b)) and (a + b + c == 1000):
                print a, b, c
