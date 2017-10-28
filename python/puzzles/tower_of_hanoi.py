#!/usr/bin/python3.5

import functools
import random


@functools.total_ordering
class disk(object):

    def __init__(self, weight):
        self._w = weight

    def __repr__(self):
        return "Disk weight %d" % self._w

    def __str__(self):
        return self.__repr__(self)

    def __eq__(self, obj):
        assert isinstance(obj, disk)
        return self._w == obj._w

    def __lt__(self, obj):
        assert isinstance(obj, disk)
        return self._w < obj._w


def tower_of_hanoi(n):
    tower0 = []
    tower1 = []
    tower2 = []

    for i in reversed(range(1, n + 1)):
        tower0.append(disk(random.randint(0, 100)))

    tower0.sort(reverse=True)

    for i in reversed(range(1, n + 1)):
        while len(tower0) > 0:
            tower1.append(tower0.pop())
        assert len(tower1) == i,\
            "tower1 len %d expected %d" % (len(tower1), i - 1)

        tower2.append(tower1.pop())

        assert len(tower0) == 0
        while len(tower1) > 0:
            tower0.append(tower1.pop())

        assert len(tower0) == i - 1
        assert len(tower1) == 0

    print(tower2)


if __name__ == "__main__":
    tower_of_hanoi(5)
