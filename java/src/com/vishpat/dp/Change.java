package com.vishpat.dp;

import java.util.Arrays;
import java.util.ArrayList;

class CoinChange {
   
    private final int[] coins = {1, 5, 10, 21, 25}; 

    public CoinChange()
    {
        Arrays.sort(coins);
    }

    public ArrayList<int> solve(int change)
    {
        int i = 0;

        for (c : this.coins) {
            minCoins[i].add(c);
            minCoins[i].addAll(solve(change - c);
        }
    }

    public static void main(String[] args) {

    }
}
