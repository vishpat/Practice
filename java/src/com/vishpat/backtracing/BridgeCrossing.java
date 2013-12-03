// Solved SRM 146 Div 2 (3rd problem)

package com.vishpat.backtracing;

import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

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
                    Set<Integer> childLeft = new HashSet<Integer>();
                    childLeft.addAll(this.left);
                    childLeft.remove(left[i]);
                    childLeft.remove(left[j]);

                    Set<Integer> childRight = new HashSet<Integer>();
                    childRight.addAll(this.right);
                    childRight.add(left[i]);
                    childRight.add(left[j]);
     
                    int childCost = this.cost + Math.max(left[i], left[j]);
                    Node childNode = new Node(this.times, childLeft, childRight, 
                                flashLightPos.RIGHT, childCost);
                    children.add(childNode);
                }
            }
        } else {
           for (Integer i : this.right) {
                Set<Integer> childLeft = new HashSet<Integer>();
                childLeft.addAll(this.left);
                childLeft.add(i);
                
                Set<Integer> childRight = new HashSet<Integer>();
                childRight.addAll(this.right);
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

    public void printConfiguration() {

        System.out.format("Flashlight %s: Left : {", 
                    this.pos == flashLightPos.LEFT ? "LEFT": "RIGHT");
        
        for (int i : this.left) {
            System.out.format("%d,", i);
        }

        System.out.print("} Right: {");
        for (int i : this.right) {
            System.out.format("%d, ", i);
        }

        System.out.print("}\n");
    }
}


class BridgeCrossing {
    public static void main(String[] args) {
        int times[] = {1, 2, 3, 50, 99, 100};
        Stack<Node> stack = new Stack<Node>();    
        Set<Integer> left = new HashSet<Integer>();
        int minCost = Integer.MAX_VALUE;

        if (times.length == 1) {
            System.out.format("%d\n", times[0]);
            return;
        }

        for (int i : times) {
            left.add(i);
        }

        Set<Integer> right = new HashSet<Integer>();
        
        Node n = new Node(times, left, right, flashLightPos.LEFT, 0);
        stack.push(n);
        
        while (!stack.empty()) {
            Node childNode = stack.pop();
            if (childNode.isLeaf()) {
                
                if (childNode.cost < minCost) {
                    minCost = childNode.cost;
                }

            } else {
                for (Node cn : childNode.getChildren()) {
                    stack.push(cn);
                }
            }
        }
        
        System.out.format("%d\n", minCost);
    }
}
