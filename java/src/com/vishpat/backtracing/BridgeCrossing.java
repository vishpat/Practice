package com.vishpat.backtracing;

import java.util.HashSet;
import java.util.Set;

enum flashLightPos {
    LEFT, RIGHT;
};

class Node {
    Set<Integer> left;
    Set<Integer> right;
    flashLightPos pos;
    int times[];
    int cost;

    public Node(int times[], 
                Set<Integer> left, 
                Set<Integer> right, 
                flashLightPos pos,
                int cost) {
        this.times = times;
        this.left = left;
        this.right = right;
        this.pos = pos;
        this.cost = cost;
    }
    
    public Set<Node> getChildren() {
        Set<Node> children = new HashSet<Node>();
        
        if (this.pos == flashLightPos.LEFT) {
            Integer left[] = this.left.toArray(new Integer[this.left.size()]);
            for (int i = 0; i < left.length; i++) {
                for (int j = i + 1; j < left.length; j++) {

                }
            }
        } else {
           for (Integer i : this.right) {
                Set<Integer> childLeft = new HashSet<Integer>();
                childLeft.addAll(this.left);
                childLeft.add(i);
                
                Set<Integer> childRight = new HashSet<Integer>();
                childRight.addAll(this.left);
                childRight.remove(i);
 
                int childCost = this.cost + i;
                Node childNode = new Node(this.times, childLeft, childRight, 
                            flashLightPos.LEFT, childCost);
                children.add(childNode);
           }
        }
        
        return children;
    }

    public boolean isLeaf() {
        return right.size() == times.length;
    }

}


class BridgeCrossing {
    public static void main(String[] args) {
        int times[] = {1, 2, 5, 10, 20};
        
        for (int i = 0; i < times.length; i++) {
            for (int j = i + 1; j < times.length; j++) {
                System.out.format("(%d, %d) ", times[i], times[j]);
            }
        }

    }
}
