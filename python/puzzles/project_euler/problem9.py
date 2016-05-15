#!/usr/bin/env python

import sys

if __name__ == "__main__":
    for x in range(1, 1001):
        for y in range(1, 1001 - x):
            for z in range(1, 1001 - (x + y)):
                    if x*x + y*y == z*z and (x + y + z == 1000):
                        print x, y, z
                        sys.exit(1)
