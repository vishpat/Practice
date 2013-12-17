package com.vishpat.dp;

enum sign {
    positive,
    negative
};

class zigzag {

    public static int longestZigZag(int[] sequence) {
        int max_seq = 0;
        if (sequence.length < 2) {
            return 1;
        }

        int dp[] = new int[sequence.length];
        for (int i = 2; i < sequence.length; i++) {
            for (int j = i - 1; j > 0; j--) {
                if (((sequence[i] > sequence[j] &&
                    sequence[j] < sequence[j - 1]) || 
                    (sequence[i] < sequence[j] &&
                    sequence[j] > sequence[j - 1])) &&
                    dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;

                    if (dp[i] > max_seq) {
                        max_seq = dp[i];
                    }
                } 
            }
        }

        return max_seq;
    }

    public static void main(String[] args) 
    {
        int[] sequence = {1,7,4,9,2,5};
        System.out.format("%d\n", longestZigZag(sequence));
    }
}
