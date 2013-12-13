package com.vishpat.dp;

import java.util.Arrays;
import java.util.ArrayList;

class CoinChange {
   
    private final int[] coins = {1,5,10,21,25}; 

    public void solve()  
    {
        int sum = 63;
        int[] sum_arr = new int[sum + 1];
       
        for (int i = 0; i < sum + 1 ; i++) {
            sum_arr[i] = Integer.MAX_VALUE; 
        }
        
        sum_arr[0] = 0;
        for (int i = 1; i < sum + 1; i++) {
            
            for (int c : coins) {
                if (i == c) {
                    sum_arr[i] = 1;
                    break;
                }

                if (i >= c &&  sum_arr[i] > sum_arr[i - c] + 1) {
                    sum_arr[i] = sum_arr[i - c] + 1;
                }
            }
       }

       System.out.format("%d\n", sum_arr[sum]);
    }

    public static void main(String[] args) {
        CoinChange c = new CoinChange();
        c.solve();
    }
}
