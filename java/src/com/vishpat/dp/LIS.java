package com.vishpat.dp;

public class LIS {
    
    static void solve() {
        int sequence[] = {15, 14, 1, 3, 4, 2, 5, 1, 7, 10, 3, 11, 12, 4, 14};
        int seq_len_arr[] = new int[sequence.length];
        int prev[] = new int[sequence.length];
        int max_idx = 0;
        int max_seq_len = 0;
      
        int i = 0;
        while (sequence[i] > sequence[i + 1]) {
            i++;
        }
        seq_len_arr[i] = 1;

        for (i = 1; i < sequence.length; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (sequence[i] > sequence[j] &&
                    seq_len_arr[i] < seq_len_arr[j]) {
                    prev[i] = j;
                    seq_len_arr[i] = seq_len_arr[j] + 1;
                    if (max_seq_len < seq_len_arr[i]) {
                        max_seq_len = seq_len_arr[i];
                        max_idx = i;
                    }
                }
            }
        }
    
        i = max_idx;
        while (prev[i] != i) {
            System.out.format("%d\n", sequence[i]);
            i = prev[i];
        }

        System.out.format("Max sequence len = %d\n", max_seq_len);
    }

    public static void main(String[] args) {
        solve();
    }
}
