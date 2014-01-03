package com.vishpat.dp;

public class LIS {
    
    static void solve() {
        int max = -1;
        int sequence[] = { 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
        int dp[] = new int[sequence.length];
        int i, j;

        i = 0;
        while (sequence[i] > sequence[i + 1]) {
            i++;
        }

        dp[i] = 1;
        for (i = 1; i < sequence.length; i++) {
            for (j = i - 1; j >= 0; j--) {
                if (sequence[i] > sequence[j] && 
                    dp[i] < dp[j]) {
                    dp[i] = dp[j] + 1;
                    if (max < dp[i]) {
                        max = dp[i];
                    }
                }
            }
        }

        System.out.format("Max sequence len = %d\n", max + 1);
    }

    public static void main(String[] args) {
        solve();
    }
}
