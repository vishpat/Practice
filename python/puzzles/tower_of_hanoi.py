#!/usr/bin/python


def tower_of_hanoi(n):
    tower0 = []
    tower1 = []
    tower2 = []

    for i in reversed(range(1, n + 1)):
        tower0.append(i)

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
