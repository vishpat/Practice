#!/usr/bin/python


class node(object):

    def __init__(self, value):
        self._value = value
        self._left = None
        self._right = None
        self._height = 0

    def add_child(self, n):

        if n < self._value:
            if self._left is None:
                self._left = node(n)
            else:
                self._left.add_child(n)
        else:
            if self._right is None:
                self._right = node(n)
            else:
                self._right.add_child(n)

    def dfs_visit(self, f):
        if self._left is not None:
            self._left.dfs_visit(f)
        if self._right is not None:
            self._right.dfs_visit(f)

        f(self._value)


def print_func(x):
    print(x)


if __name__ == "__main__":
    root = node(2)
    for e in [8, 5, 6, 3, 10, 12, 14]:
        root.add_child(e)

    root.dfs_visit(print_func)
