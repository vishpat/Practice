#!/usr/bin/env python

import functools
import itertools
import operator
import collections


def factors(number):
    primes = list()
    i = 2
    num = number
    while i < num:
        if num % i == 0:
            primes.append(i)
            num = num / i
            i = 2
        else:
            i += 1
    primes.append(i)
    return primes


def lcm(numbers):
    global_factors = collections.defaultdict(int)

    for n in numbers:
        for x, y in itertools.groupby(factors(n)):
            facts = list(y)
            count = len(facts)

            if count > global_factors[x]:
                global_factors[x] = count

    val = 1
    for k, v in global_factors.iteritems():
        val *= pow(k, v)

    return val


if __name__ == "__main__":
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    prime_prod = functools.reduce(operator.mul, primes, 1)
    non_primes = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, prime_prod]
    print lcm(non_primes)
