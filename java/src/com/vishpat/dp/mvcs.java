package com.vishpat.dp;

public class mvcs {

    static void solve() {
        int sequence[] = {-15, 29, -36, 3, -22, 11, 19, -5};
        int sum_idx[] = new int[sequence.length];
        int sum[] = new int[sequence.length];
        
        int max_sum = 0;
        int max_sum_idx = 0;

        sum[0] = sequence[0];
        for (int i = 1; i < sequence.length; i++) {
            
            if (sum[i - 1] + sequence[i] > sequence[i]) {
                sum[i] = sum[i - 1] + sequence[i];
                sum_idx[i] = i - 1;
            } else {
                sum[i] = sequence[i];
                sum_idx[i] = i;
            }
            
            if (sum[i] > max_sum) {
                max_sum = sum[i];
                max_sum_idx = i;
            }
        }
        
        int min_sum_idx = max_sum_idx;
        while (sum_idx[min_sum_idx] != min_sum_idx) {
           min_sum_idx--; 
        }


        System.out.format("(%d, %d) :%d\n", 
                    min_sum_idx, max_sum_idx, max_sum);
    }

    public static void main(String[] args) {
        solve();
    }
}
