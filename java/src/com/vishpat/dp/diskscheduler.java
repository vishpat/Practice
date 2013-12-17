package com.vishpat.dp;

import java.util.HashSet;
import java.util.Arrays;

class DiskScheduler {

    static int 
    seeks(int start_sec, int next_req, HashSet<Integer> requests)
    {
        if (requests.size() == 0) {
            return Math.abs(start_sec - next_req);
        }

        int min_seeks = Integer.MAX_VALUE;
        int cur_seeks = 0;

        for (int req: requests) {
            
            HashSet<Integer> remaining = new HashSet<Integer>();
            remaining.addAll(requests);
            remaining.remove(req);
            
            cur_seeks = seeks(next_req, req, remaining);

            if (cur_seeks < min_seeks) {
                min_seeks = cur_seeks;
            }
        }
        
        return Math.abs(start_sec - next_req) + min_seeks;
    }

    static void solve()
    {
        int min_seeks = Integer.MAX_VALUE;
        int start_sec = 140;
        int requests[] = {100, 50, 190};
        int cur_seeks = 0;

        for (int req: requests) {
            
            HashSet<Integer> remaining = new HashSet<Integer>();

            for (int req2: requests) {
                if (req2 == req) {
                    continue;
                }
                remaining.add(new Integer(req2));
            }
            
            cur_seeks = seeks(start_sec, req, remaining);

            if (cur_seeks < min_seeks) {
                min_seeks = cur_seeks;
            }
        }
        
        System.out.format("%d\n", min_seeks);

    }

    public static void main(String[] args) 
    {
        solve();        
    }
}
