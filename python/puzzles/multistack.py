#!/usr/bin/python


class MultiStack(object):

    def __init__(self, n):
        self._n = n
        self._stacks = []

    def push(self, a):

        if len(self._stacks) == 0:
            self._stacks.append([])
            self.cur_idx = 0

        stack = self._stacks[self.cur_idx]
        if len(stack) == self._n:
            stack = []
            self._stacks.append(stack)
            self.cur_idx = self.cur_idx + 1
        stack.append(a)

    def empty(self):
        return len(self._stacks) == 0

    def pop(self):
        if len(self._stacks) == 0:
            return None

        stack = self._stacks[self.cur_idx]
        e = stack.pop()
        if len(stack) == 0:
            self._stacks.remove(stack)
            self.cur_idx = self.cur_idx - 1

        return e

    def __str__(self):
        return str(self._stacks)


if __name__ == "__main__":
    ms = MultiStack(3)

    for i in range(0, 10):
        ms.push(i)

    while not ms.empty():
        print(ms.pop())
