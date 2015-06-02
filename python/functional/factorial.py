__author__ = 'vishpat'

def fact(n):
    return reduce(lambda x, y: x*y, [x for x in xrange(1, n + 1)])

if __name__ == "__main__":
    print fact(5)