
def num_ways(data):

    if len(data) == 0:
        return 0

    if len(data) == 1:
        return 1

    if len(data) == 2:
        if int(data) <= 26:
            return 2
        else:
            return 1

    return 1 + num_ways(data[1:])


if __name__ == "__main__":
    print(num_ways("2222"))


