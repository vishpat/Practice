package com.vishpat.backtracing;

import java.util.Set;
import java.util.TreeSet;

class Pos {
    public int x;
    public int y;
}

class Node {
    private Set<Pos> positions;
    private Set<Node> children;
    private int board_size;

    public Node(Set<Pos> positions, int board_size) {
        this.positions = positions;
        this.board_size = board_size;
    }
    
    public boolean isLeaf() {
        return this.board_size == this.positions.size();
    }

    public Set<Pos> getValidPositions() {
        
        Set<Pos> validPositions = new TreeSet<Pos>();
        int xpos = 0;
        int ypos = 0;
        
        for (Pos pos: positions) {
            for (int i = 0; i < this.board_size; i++) {
                for (int j = 0; j < this.board_size; j++) {
                    if (i == pos.x || j == pos.y) {
                        continue;
                    }
                }
            }    
        }
        
        return validPositions;
    }
}

class Queens {
    public static void main(String[] args) {
        System.out.println("Hello java!!!\n");
    }
}
