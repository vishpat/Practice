#!/usr/bin/env python

import re


def mac_generator():
    for mac in range(1099511627777, 1099511627777 + 1001):
        mac_str = "%012x" % mac
        yield ':'.join(re.findall('..?', mac_str))


if __name__ == "__main__":
    for mac in mac_generator():
        print mac
