from itertools import permutations


class Box:
    def __init__(self, width, length, height):
        self.width = width
        self.length = length
        self.height = height

    def __lt__(self, other):
        return self.width < other.width and self.length < other.length

    def __repr__(self):
        return f"({self.width},{self.length},{self.height})"


def max_height(boxes):
    boxes = sorted(boxes, key=lambda box: (box.width, box.length))
    total_max_height = 0
    max_height = [box.height for box in boxes]
    for i in range(0, len(boxes)):
        for j in range(0, i):
            if i == j:
                continue
            new_max_height = max_height[j] + boxes[i].height

            if boxes[j] < boxes[i] and new_max_height > max_height[i]:
                max_height[i] = new_max_height
                total_max_height = (
                    max_height[i]
                    if max_height[i] > total_max_height
                    else total_max_height
                )

    return total_max_height


if __name__ == "__main__":
    boxes = [(4, 2, 5), (3, 1, 6), (3, 2, 1), (6, 3, 8)]
    all_possible_arrangements = list()
    for box_dimension in boxes:
        for dimensions in permutations(box_dimension):
            all_possible_arrangements.append(
                Box(dimensions[0], dimensions[1], dimensions[2])
            )
    print(max_height(all_possible_arrangements))
