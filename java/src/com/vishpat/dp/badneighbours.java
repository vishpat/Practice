package com.vishpat.dp;

class BadNeighbours {

    public static int getMaxDonations(int[] donations)
    {
        int max = -1;
        int maxDonations[] = new int[donations.length];

        if (donations.length == 1) {
            return donations[0];
        }

        if (donations.length == 2) {
            return Math.max(donations[0], donations[1]);
        }

        maxDonations[0] = donations[0];
        maxDonations[1] = donations[1];
        
        for (int i = 2; i < donations.length - (donations.length % 2); i++) {
            maxDonations[i] = maxDonations[i - 2] + donations[i];
            if (max < maxDonations[i]) {
                max = maxDonations[i];
            }
        }

        return max;
    }

    public static void main(String[] args) {
//       int donations[] = {94,40,49,65,21,21,106,80,92,81,679,4,61,6,237,12,72,74,29,95,265,35,47,1,61,397,52,72,37,51,1,81,45,435,7,36,57,86,81,72};
        int donations[] = {1,2,3,4,5,1,2,3,4,5};
        System.out.format("%d, %d\n", 
                            donations.length, 
                            getMaxDonations(donations));
    }
}
