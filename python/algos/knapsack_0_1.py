from itertools import combinations


class Item:
    def __init__(self, w, v):
        self.w = w
        self.v = v

    def __lt__(self, other):
        return self.w < other.w

    def __repr__(self):
        return f"({self.w},{self.v})"


max_knapsack_val = -1
max_knapsack = {}


def max_value_child(W, items, indices):
    indices.sort()
    key = tuple(indices)
    if key in max_knapsack:
        return max_knapsack[key]

    global max_knapsack_val
    max_val = -1
    if len(indices) == 1:
        idx = indices[0]
        item = items[idx]
        return item.v

    for i in range(0, len(indices)):
        sub_indices = []
        for j in range(0, len(indices)):
            sub_indices.append(indices[j]) if i != j else None
        sub_indices.sort()

        item = items[indices[i]]
        v = max_value_child(W, items, sub_indices)
        w = sum(map(lambda idx: items[idx].w, sub_indices))

        if w + item.w <= W:
            max_val = item.v + v

        if max_val > max_knapsack_val:
            max_knapsack_val = max_val

    max_knapsack[key] = max_val
    return max_val


def max_value(W, items):
    max_value_child(W, items, [i for i in range(0, len(items))])
    return max_knapsack_val


if __name__ == "__main__":
    print(
        max_value(
            10,
            [
                Item(1, 20),
                Item(2, 5),
                Item(3, 10),
                Item(8, 40),
                Item(7, 15),
                Item(4, 25),
            ],
        )
    )
