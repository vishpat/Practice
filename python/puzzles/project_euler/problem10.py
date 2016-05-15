#!/usr/bin/env python


if __name__ == "__main__":
    k = 2000000 + 1
    prime_list = [True for x in range(0, k)]

    prime_list[0] = False
    prime_list[1] = False
    prime_list[2] = True

    num = 2
    while num < k:

        idx = 1
        while idx*num < k:
            if idx != 1:
                prime_list[idx*num] = False
            idx += 1

        x = num + 1
        next_prime_found = False
        while x < k:
            if prime_list[x] is True:
                num = x
                next_prime_found = True
                break
            x += 1

        if next_prime_found is False:
            break

    final_sum = 0
    for idx, val in enumerate(prime_list):
        if val is True:
            final_sum += idx

    print final_sum
