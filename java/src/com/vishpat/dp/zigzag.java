package com.vishpat.dp;

enum sign {
    positive,
    negative
};

class zigzag {

    public static int longestZigZag(int[] sequence) {

        if (sequence.length < 2) {
            return 1;
        }

        sign current_sign = (sequence[1] > sequence[0]) ? 
                            sign.positive : sign.negative;

        int dp[] = new int[sequence.length];
        for (int i = 1; i < sequence.length; i++) {
        
        }

        return 0;
    }

    public static void main(String[] args) 
    {
        int[] sequence = {1,7,4,9,2,5};
        longestZigZag(sequence);
    }
}
