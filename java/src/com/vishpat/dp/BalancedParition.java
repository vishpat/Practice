package com.vishpat.dp;

import java.util.ArrayList;

class BalancedParition {
   
    static void printSet(String setName, ArrayList<Integer> elements)
    {
        System.out.format("%s: ", setName);

        for (int v: elements) {
            System.out.format("%d ", v);
        }

        System.out.format("\n");
    }

    static int sum(ArrayList<Integer> elements, int element)
    {
        int total = 0;
        for (int v : elements) {
            total += v;
        }

        return total + element;
    }

    static void solve(int[] values) 
    {
        ArrayList <Integer>s1 = new ArrayList<Integer>();
        ArrayList <Integer>s2 = new ArrayList<Integer>();
        
        s1.add(values[0]);
        s2.add(values[1]);

        for (int i = 2; i < values.length; i++) {
            int element = values[i];

            int sum1 = sum(s1, element);
            int s1_sum = sum(s1, 0);
            
            int sum2 = sum(s2, element);
            int s2_sum = sum(s2, 0);

            if ((sum1 - s2_sum) < (sum2 - s1_sum)) {
                s1.add(element);
            } else {
                s2.add(element);
            }
        }
        
        printSet("S1", s1);
        printSet("S2", s2);
    }

    public static void main(String[] args) 
    {
        int values[] = {2, 10, 3, 8, 5, 7, 9, 5, 3, 2};
        solve(values);
    }
}
