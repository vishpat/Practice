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
    max_stack_height = [box.height for box in boxes]
    for i in range(0, len(boxes)):
        for j in range(0, i):
            if boxes[j] < boxes[i]: 
                new_max_stack_height = max_stack_height[j] + boxes[i].height
                if new_max_stack_height > max_stack_height[i]:
                    max_stack_height[i] = new_max_stack_height

    return max(max_stack_height)


if __name__ == "__main__":
    boxes = [(4, 2, 5), (3, 1, 6), (3, 2, 1), (6, 3, 8)]
    all_possible_arrangements = list()
    for box_dimension in boxes:
        for dimensions in permutations(box_dimension):
            all_possible_arrangements.append(
                Box(dimensions[0], dimensions[1], dimensions[2])
            )
    print(max_height(all_possible_arrangements))
