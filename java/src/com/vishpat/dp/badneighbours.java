package com.vishpat.dp;

// TopCoder TCCC 04, Round 04

class BadNeighbours {

    public static int getMaxDonations(int[] donations)
    {
        int total = donations.length;
        int dp[] = new int[total];
        int max = -1;
        int i = 0;

        dp[0] = donations[0];
        dp[1] = donations[1];
        dp[2] = donations[2] + donations[0];
        
        for (i = 3; i < donations.length - 1; i++) {
            dp[i] = donations[i] + Math.max(dp[i - 2], dp[i - 3]);
            max = max > dp[i] ? max : dp[i];
        }

        dp[1] = donations[1];
        dp[2] = donations[2];
        dp[3] = donations[3] + donations[1];

        for (i = 4; i < donations.length; i++) {
            dp[i] = donations[i] + Math.max(dp[i - 2], dp[i - 3]);
            max = max > dp[i] ? max : dp[i];
        }


        return max;
    }

    public static void main(String[] args) {
        int donations[] = {94,40,49,65,21,21,106,80,92,81,679,4,61,6,237,12,72,74,29,95,265,35,47,1,61,397,52,72,37,51,1,81,45,435,7,36,57,86,81,72};
        System.out.format("%d\n", 
                            getMaxDonations(donations));
    }
}
