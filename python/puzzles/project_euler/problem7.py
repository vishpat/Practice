#!/usr/bin/env python


def is_prime(prime_list, num):

    for prime in prime_list:
        if num % prime == 0:
            return False

    return True


if __name__ == "__main__":
    prime_list = [2, 3, 5, 7, 11, 13]
    prime_count = len(prime_list)
    num = 13
    while num >= 13:

        if is_prime(prime_list, num):
            prime_list.append(num)

        if len(prime_list) == 10001:
            print prime_list[-1]
            break

        num += 1
