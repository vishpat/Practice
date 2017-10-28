#!/usr/bin/python

from collections import deque


def braces(count):
    queue = deque()
    queue.append("()")
    braces = []

    for idx in range(0, count):
        temp = []
        while queue:
            e = queue.popleft()
            temp.append(e)
        
        if idx == count - 1:
            return temp

        for e in temp:
            braces.append(e)
            queue.append("(" + e + ")")
            queue.append("()" + e)

    return []


if __name__ == "__main__":
    print(braces(4))
