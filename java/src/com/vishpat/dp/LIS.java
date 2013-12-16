package com.vishpat.dp;

public class LIS {
    
    static int solve() {
        int sequence[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
        int length_arr[] = new int[sequence.length];
        int parent_arr[] = new int[sequence.length];
        int max_idx = 0;
        int max_seq_len = 0;
        
        length_arr[0] = 0;
        for (int i = 1; i < sequence.length; i++) {
            if (sequence[i] > sequence[i - 1]) {
                length_arr[i] = length_arr[i - 1] + 1;
                parent_arr[i] = i - 1;
                if (max_seq_len < length_arr[i]) {
                    max_seq_len = length_arr[i];
                    max_idx = i;
                }
            } else {
                length_arr[i] = length_arr[i - 1];
                parent_arr[i] = parent_arr[i - 1];
            }
        }
    }

    public static void main(String[] args) {
        solve();
    }
}
