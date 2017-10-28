#!/usr/bin/python
 

def magic_idx(arr, start_idx, end_idx):

    if arr[start_idx] == start_idx:
        return start_idx

    if arr[end_idx] == end_idx:
        return end_idx

    if (start_idx < end_idx):
        mid_idx = (start_idx + end_idx) / 2
        left_idx = magic_idx(arr, start_idx, mid_idx)
        if left_idx is not None:
            return left_idx

        right_idx = magic_idx(arr, mid_idx + 1, end_idx)
        if right_idx is not None:
            return right_idx

    return None


if __name__ == "__main__":
    arr = [-23, -4, 2, 19, 56]
    arr = [-10, -3, 0, 2, 4, 8]
    print(magic_idx(arr, 0, len(arr) - 1))
