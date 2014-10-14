#!/usr/bin/env python

import itertools
import sys

numbers = [3, 4, 5, -3, 100, 1, 89, 54, 23, 20]

def main():
    min_diff = sys.maxint
    min_s1 = None
    min_s2 = None
    
    subsets = itertools.combinations(numbers, len(numbers)/2)
    for s1 in subsets:
        s2 = list(numbers)
        for item in s1:
            s2.remove(item)
        diff = abs(sum(s1) - sum(s2))
        if diff < min_diff:
            min_diff = diff
            min_s1 = list(s1)
            min_s2 = list(s2)

    print min_diff
    print min_s1
    print min_s2

if __name__ == "__main__":
    main()
