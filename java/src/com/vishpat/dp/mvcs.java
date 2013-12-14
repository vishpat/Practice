package com.vishpat.dp;

public class mvcs {

    static void solve() {
        int sequence[] = {-15, 29, -36, 3, -22, 11, 19, -5};
        int sum[] = new int[sequence.length];
        int max_sum = 0;
        sum[0] = sequence[0];
        for (int i = 1; i < sequence.length; i++) {
            
            if (sum[i - 1] + sequence[i] > sequence[i]) {
                sum[i] = sum[i - 1] + sequence[i];
            } else {
                sum[i] = sequence[i];
            }
            
            if (sum[i] > max_sum) {
                max_sum = sum[i];
            }
        }

        System.out.format("%d\n", max_sum);
    }

    public static void main(String[] args) {
        solve();
    }
}
