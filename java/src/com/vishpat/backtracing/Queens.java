package com.vishpat.backtracing;
import java.util.Vector;

class Pos {
    public int x;
    public int y;
}

class Node {
    private Vector<Pos> positions;
    private Vector<Node> children;
    private int board_size;

    public Node(Vector<Pos> positions, int board_size) {
        this.positions = positions;
        this.board_size = board_size;
    }
    
    public boolean isLeaf() {
        return this.board_size == this.positions.size();
    }

    public Vector<Pos> getValidPositions(Pos pos) {
        
        Vector<Pos> validPositions = new Vector<Pos>();
        int xpos = 0;
        int ypos = 0;
        
        for (int i = 0; i < this.board_size; i++) {
            for (int j = 0; j < this.board_size; j++) {
                
                if (i == pos.x || j == pos.y) {
                    continue;
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
