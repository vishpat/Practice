package com.vishpat.dp;

public class LIS {
    
    static void solve() {
        int sequence[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
        int seq_len_arr[] = new int[sequence.length];
        int parent_arr[] = new int[sequence.length];
        int max_idx = 0;
        int max_seq_len = 0;
       
        for (int i = 0; i < sequence.length; i++) {
            seq_len_arr[i] = 1;
        }

        for (int i = 1; i < sequence.length; i++) {
            for (int j = 0; j < i; j++) {
                if (sequence[i] > sequence[j]) {
                    seq_len_arr[i] += 1;

                    if (max_seq_len < seq_len_arr[i]) {
                        max_seq_len = seq_len_arr[i];
                    }
                }
            }
        }

        System.out.format("%d\n", max_seq_len);
    }

    public static void main(String[] args) {
        solve();
    }
}
