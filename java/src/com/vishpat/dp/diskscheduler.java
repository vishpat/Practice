package com.vishpat.dp;

import java.util.HashSet;
import java.util.Arrays;

class DiskScheduler {
    static int start_sec = 140;
    static int requests[] = {100, 50, 190};
    static int min[][] = new int[requests.length][requests.length];

    static int min_seeks(int i, int j)
    {
        if (min[i][j] == -1) {
            min[i][j] = min[j][i] = Math.abs(requests[i] - requests[j]);
        }
        
        return min[i][j];
    }

    static void solve()
    {   
        int min_dist = Integer.MAX_VALUE;
        int cur_seeks = 0;
 
        for (int i = 0; i < requests.length; i++) {
            for (int j = 0; j < requests.length; j++) {
                
                if (i == j) {
                    continue;
                }

                min[i][j] = -1;
            }
        }

        for (int i = 0; i < requests.length; i++) {
            for (int j = 0; j < requests.length; j++) {
                
                if (i == j) {
                    continue;
                }

                int dist = Math.abs(start_sec - requests[i]) +
                            min_seeks(i, j);

                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }

        System.out.format("%d\n", min_dist);
    }

    public static void main(String[] args) 
    {
        solve();        
    }
}
