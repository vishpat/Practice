#!/usr/bin/env python

import sys
from shortest_path import shortest_path
from shortest_path import shortest_distance


safe = 0
harmful = 1
deadly = sys.maxint 
size = 50 

class Escape:
    def lowest(self, harmful_areas, deadly_areas):
        
        matrix = list()
        for row in range(0, size):
            row = list()
            matrix.append(row)

            for col in range(0, size):
                row.append(safe)
        
        for h in harmful_areas:
            x1, y1, x2, y2 = [int(n) for n in h.split()]
            for i in range(min(x1, x2), max(x1, x2)):
                for j in range(min(y1, y2), max(y1, y2)):
                    matrix[i][j] = harmful

        for d in deadly_areas:
            x1, y1, x2, y2 = [int(n) for n in d.split()]
            for i in range(min(x1, x2), max(x1, x2)):
                for j in range(min(y1, y2), max(y1, y2)):
                    matrix[i][j] = deadly 

        graph = {}
        for i in range(0, size):
            for j in range(0, size):
                state = matrix[i][j]
                
                if (state == deadly):
                    continue
                
                node = (i, j)
                neighbours = list()

                if i > 0 and matrix[i - 1][j] != deadly :
                    neighbours.append(((i - 1, j), matrix[i - 1][j]))
                
                if i < size - 1  and matrix[i + 1][j] != deadly :
                    neighbours.append(((i + 1, j), matrix[i + 1][j]))
 
                if j > 0 and matrix[i][j - 1] != deadly :
                    neighbours.append(((i, j - 1), matrix[i][j - 1]))

                if j < size - 1 and matrix[i][j + 1] != deadly :
                    neighbours.append(((i, j + 1), matrix[i][j + 1]))

                graph[node] = neighbours

        min_path = shortest_path(graph, (0, 0), (size, size))
        min_dist = shortest_distance(graph, (0, 0), (size, size))

        print str(min_dist)


if __name__ == "__main__":
    esc = Escape()
    esc.lowest(["50 0 0 50"], ["0 0 0 0"])
