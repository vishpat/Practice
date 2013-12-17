package com.vishpat.dp;

class zigzag {

    public static int longestZigZag(int[] sequence) {
        int max_seq = 0;
        if (sequence.length < 2) {
            return 1;
        }

        int dp[] = new int[sequence.length];
        
        dp[0] = 1;
        dp[1] = 2;

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
        int[] sequence = {374, 40, 854, 203, 203, 156, 362, 279, 812, 955, 
        600, 947, 978, 46, 100, 953, 670, 862, 568, 188, 
        67, 669, 810, 704, 52, 861, 49, 640, 370, 908, 
        477, 245, 413, 109, 659, 401, 483, 308, 609, 120, 
        249, 22, 176, 279, 23, 22, 617, 462, 459, 244};
        System.out.format("%d\n", longestZigZag(sequence));
    }
}
