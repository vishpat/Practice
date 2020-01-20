#!/usr/bin/env python

import sys


def problem_sum(input, s):
    start_idx = 0
    wsize = 1
    wsum = input[0]

    while start_idx + wsize <= len(input):
        
        if wsum == s:
            return start_idx + 1, start_idx + wsize

        if wsum > s:
            wsum -= input[start_idx]
            start_idx += 1
            wsize -= 1
        else:
            wsize += 1
            if start_idx + wsize > len(input):
                return -1, -1

            wsum += input[start_idx + wsize - 1]

    return -1, -1



if __name__ == "__main__":
    test_input = []
    for line in sys.stdin:
        test_input.append(line)

    test_case_count = int(test_input[0])
    for lineno in range(1, 2 * test_case_count, 2):
        array_size, _sum = test_input[lineno].strip().split()
        _sum = int(_sum)
        problem_input = map(lambda x: int(x), test_input[lineno + 1].strip().split())
        start, end = problem_sum(list(problem_input), _sum)
        if start == -1:
            print(-1)
        else:
            print(start, end)
