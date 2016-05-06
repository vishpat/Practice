#!/usr/bin/env python


def is_palindrome(num):
    return str(num) == str(num)[::-1]


if __name__ == "__main__":
    max_palindrome = 0
    for x in reversed(range(100, 1000)):
        y = 999
        while y >= x:
            if is_palindrome(x*y) is True and x*y > max_palindrome:
                max_palindrome = x*y
            y -= 1
    print max_palindrome
