package com.vishpat.backtracing;

import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;

class Pos {
    public int x;
    public int y;

    public Pos(int x, int y) {
        this.x = x;
        this.y = y;
    }
}

class Node {

    private Set<Pos> configuration;
    private Set<Node> children;
    private int board_size;

    public Node(Set<Pos> configuration, int board_size) {
        this.configuration = configuration;
        this.board_size = board_size;
    }

    public Set<Node> getChildren() {
        TreeSet<Node> children = new TreeSet<Node>();
        Set<Pos> positions = this.getNextValidPositions();

        for (Pos p: positions){
            TreeSet<Pos> nextConfiguration = new TreeSet<Pos>();
            nextConfiguration.addAll(this.configuration);
            nextConfiguration.add(p);
            Node childNode = new Node(nextConfiguration, this.board_size); 
            children.add(childNode);
        }

        return children; 
    }

    public boolean isLeaf() {
        return this.board_size == this.configuration.size();
    }

    public Set<Pos> getNextValidPositions() {
        Vector <Set<Pos>> positions = new Vector<Set<Pos>>();

        int xpos = 0;
        int ypos = 0;
        
        for (Pos pos: this.configuration) {
            Set<Pos> validPositions = new TreeSet<Pos>();
            for (int i = 0; i < this.board_size; i++) {
                for (int j = 0; j < this.board_size; j++) {
                    if (i == pos.x || j == pos.y) {
                        continue;
                    }

                    if (Math.abs(i - pos.x) == Math.abs(j - pos.y)) {
                        continue;
                    }
                    
                    Pos p = new Pos(i, j);
                    validPositions.add(p);
                }
            }
            positions.add(validPositions);
        }
      
        if (positions.size() == 0) {
            return new TreeSet<Pos>();
        }

        Set<Pos> validPosSet = positions.elementAt(0);    

        for (Set<Pos> validPositions: positions) {
            validPosSet.retainAll(validPositions); 
        }

        return validPosSet;
    }
}

class Queens {
    public static void main(String[] args) {
        System.out.println("Hello java!!!\n");
    }
}
