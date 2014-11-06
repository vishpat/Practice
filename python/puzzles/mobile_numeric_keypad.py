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


def init(N):
    for i in range(0, len(neighbors.keys())):
        cnt_bucket = {}
        cnt_bucket[0] = 0
        cnt_bucket[1] = 1
        cnt_bucket[2] = 1 + len(neighbors[i])

        neighbor_cnt[i] = cnt_bucket


def fill_neighbor_cnt(num, N):

    cnt_bucket = neighbor_cnt[num]

    if N in cnt_bucket:
        return cnt_bucket[N]
    else:
        cnt = 0
        for neighbor in neighbors[num]:
            cnt += 1 + fill_neighbor_cnt(neighbor, N - 1)
        cnt_bucket[N] = cnt
        neighbor_cnt[num] = cnt_bucket
        return cnt


def main(N):
    for i in neighbors.keys():
        total = 0
        for neighbor in neighbors[i]:
            total += 1 + fill_neighbor_cnt(neighbor, N - 1)
        neighbor_cnt[i][N] = total

    total = 0
    for x in neighbor_cnt.keys():
        total += neighbor_cnt[x][N]

    return total

if __name__ == "__main__":
    init(int(sys.argv[1]))
    print neighbor_cnt
    print main(int(sys.argv[1]))
