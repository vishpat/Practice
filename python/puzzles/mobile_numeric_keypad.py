#!/usr/bin/env python

"""Solution to mobile-numeric-keypad-problem using DP """

import sys


neighbors = {
    1: [2, 4],
    2: [1, 3, 5],
    3: [2, 6],
    4: [1, 5, 7],
    5: [2, 4, 6, 8],
    6: [3, 5, 9],
    7: [4, 8],
    8: [5, 7, 9, 0],
    9: [6, 8],
    0: [8]
}

neighbor_cnt = {}

element_cnt = 10


def init(N):
    for i in range(0, element_cnt):
        cnt_bucket = {}
        cnt_bucket[0] = 1
        cnt_bucket[1] = len(neighbors[i])

        neighbor_cnt[i] = cnt_bucket


def get_neighbor_cnt(num, N):
    cnt_bucket = neighbor_cnt[num]

    if N in cnt_bucket:
        return cnt_bucket[N]
    else:
        cnt = 0
        for neighbor in neighbors[num]:
            cnt += 1 + get_neighbor_cnt(neighbor, N - 1)
        cnt_bucket[N] = cnt
        neighbor_cnt[num] = cnt_bucket
        return cnt


def main(N):
    if N == 0:
        return 0
    elif N == 1:
        return element_cnt
    else:
        total = 0
        for i in neighbors.keys():
            total += 1 + get_neighbor_cnt(i, N - 1)
        return total

if __name__ == "__main__":
    init(int(sys.argv[1]))
    print main(int(sys.argv[1]))
