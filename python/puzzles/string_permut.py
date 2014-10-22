#!/usr/bin/env python

import sys
from itertools import permutations


def permut(soup):
    soups = list()

    if len(soup) == 2:
        a = soup[0]
        b = soup[1]
        soup2 = list(soup)
        soup2[0] = b
        soup2[1] = a
        new_soup = ''.join(soup2)
        soups.append(new_soup)
        soups.append(soup)
    else:
        prefix = soup[0]
        child_soups = permut(soup[1:])
        for child_soup in child_soups:
            new_soup = prefix + '' + child_soup
            for i in range(0, len(new_soup)):
                a = new_soup[0]
                b = new_soup[i]
                soup2 = list(new_soup)
                soup2[0] = b
                soup2[i] = a
                new_soup2 = ''.join(soup2)
                if new_soup2 not in soups:
                    soups.append(new_soup2)

    return soups


def main(soup):
    permut_list = permut(soup)

    verification_list = list()
    for p in permutations(soup):
        verification_list.append(''.join(p))

    print cmp(verification_list.sort(), permut_list.sort())

if __name__ == "__main__":
    main(sys.argv[1])
