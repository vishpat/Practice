#!/usr/bin/env python

import random


def bsearch(n_list, num, start_idx, end_idx):
    start_num = n_list[start_idx]
    end_num = n_list[end_idx]

    if (start_num == num):
        return start_idx

    if (end_num == num):
        return end_idx

    if (start_idx < end_idx):
        mid = (start_idx + end_idx) / 2
        if (n_list[mid] == num):
            return mid
        else:
            left_idx = bsearch(n_list, num, start_idx, mid)
            right_idx = bsearch(n_list, num, mid + 1, end_idx)
            if (left_idx != -1):
                return left_idx

            if (right_idx != -1):
                return right_idx

    return -1


def rbsearch(n_list, num, start_idx, end_idx):
    print(start_idx, end_idx)

    start_num = n_list[start_idx]
    end_num = n_list[end_idx]

    if (start_num == num):
        return start_idx

    if (end_num == num):
        return end_idx

    if (start_num < end_num):
        return bsearch(n_list, num, start_idx, end_idx)

    mid_idx = (start_idx + end_idx) / 2
    mid_num = n_list[mid_idx]

    if (start_num < mid_num and num > start_num and num < mid_num):
        return bsearch(n_list, num, start_idx, mid_idx)

    if (mid_num < end_num and num > mid_num and num < end_num):
        return bsearch(n_list, num, mid_idx + 1, end_idx)

    left_idx = rbsearch(n_list, num, start_idx, mid_idx)
    right_idx = rbsearch(n_list, num, mid_idx + 1, end_idx)

    if (left_idx != -1):
        return left_idx

    if (right_idx != -1):
        return right_idx

    return -1


if __name__ == "__main__":
    random.seed(10)
    rand_set = set()
    for i in range(0, 20):
        rand_set.add(random.randint(1, 100))

    rand_list = list(rand_set)
    rand_list.sort()

    for i in range(0, 3):
        a = rand_list.pop(0)
        rand_list.append(a)

    print(rand_list, rbsearch(rand_list, 60, 0, len(rand_list) - 1))
