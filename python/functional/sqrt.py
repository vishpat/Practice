__author__ = 'vishpat'

"""
Implements the Newton Rapshon's Square Roots method using Functional Programming
as described in the paper

https://www.cs.kent.ac.uk/people/staff/dat/miranda/whyfp90.pdf
"""


def repeat(a0, n):
    while True:
        a1 = (1.0 * a0 + n / a0) / 2
        yield a1
        a0 = a1


def within(eps, a0, n):
    a1 = n.next()
    return a0 if abs(a0 - a1) <= eps else within(eps, a1, n)


def functional_sqrt(n):
    eps = 0.000000001
    a0 = n / 2
    r = repeat(a0, n)
    return within(eps, a0, r)


if __name__ == "__main__":
    print functional_sqrt(12100000000000000)