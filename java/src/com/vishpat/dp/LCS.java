package com.vishpat.dp;

public class LCS {

    static void LCS(String s1, String s2) 
    {
        int[][] lcs_array = new int[s1.length()][s2.length()];
        int lcs = 0;
        
        for (int i = 0; i < s1.length(); i++) {
            for (int j = 0; j < s2.length(); j++) {
                if (s2.charAt(j) == s1.charAt(i)) {

                   int prev = 0;                
                    if (i > 0 && j > 0) {
                        prev = lcs_array[i - 1][j - 1]; 
                    }

                    lcs_array[i][j] = prev + 1;
                    
                    if (lcs_array[i][j] > lcs) {
                        lcs = lcs_array[i][j];
                    }
                }
            }
        }

        System.out.format("%d\n", lcs);
    }

    public static void main(String[] args) 
    {
        LCS(args[0], args[1]);    
    }
}
