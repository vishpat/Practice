package com.vishpat.dp;

import java.util.ArrayList;

class BalancedParition {
    
    static int sum(ArrayList<int> elements, int element)
    {
        int total = 0;
        for (int v : elements) {
            total += v;
        }

        return total + element;
    }

    static void solve(int[] values) 
    {
        Arraylist s1 = new ArrayList<int>();
        Arraylist s2 = new ArrayList<int>();
        
        s1.add(values[0]);
        s2.add(values[1]);

        for (int i = 2; i < values.length; i++) {
            int element = values[i];

            int sum1 = sum(s1, element);
            int s1_sum = sum(s1, 0);
            
            int sum2 = sum(s2, element);
            int s2_sum = sum(s2, 0);
            
        }
    }

    public static void main(String[] args) 
    {
        int values[] = {};
        solve(values);
    }
}
