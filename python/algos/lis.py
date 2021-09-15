def lis(arr):
    max_seq = [1 for _ in range(0, len(arr))]
    lcs_max_len = 1
    for i in range(0, len(arr)):
        for j in range(0, i):
            if arr[j] < arr[i] and max_seq[j] + 1 > max_seq[i]:
                max_seq[i] = max_seq[i] + 1
                lcs_max_len = max_seq[i] if max_seq[i] > lcs_max_len else lcs_max_len
    return lcs_max_len

if __name__ == "__main__":
    print(lis([50, 3, 10, 7, 40, 80]))
