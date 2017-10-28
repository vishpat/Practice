#!/usr/bin/python

import copy


def add_to_powerset(power_set, e):
    print(power_set, e)
    new_subsets = list()
    for subset in power_set:
        new_subset = copy.deepcopy(subset)
        new_subset.append(e)
        print(new_subset)
        new_subsets.append(new_subset)
    
    print(new_subsets)
    power_set.append(new_subsets)
    
    print(power_set)

if __name__ == "__main__":
    s = ["a", "b"]
    powerset = [[]]
    for e in s:
        add_to_powerset(powerset, e)

    print(powerset)
