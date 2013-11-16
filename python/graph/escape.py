#!/usr/bin/env python

class Escape:
    
    def lowest(self, deadly_areas, harmful_areas):
        
        safe = 0
        harmful = 1
        deadly = 2
        size = 500 

        matrix = list()
        for row in range(0, size):
            row = list()
            matrix.append(row)

            for col in range(0, size):
                row.append(safe)
        
        for h in harmful_areas:
            x1, y1, x2, y2 = [int(n) for n in h.split()]
            for i in range(x1, x2 + 1):
                for j in range(y1, y2 + 1):
                    matrix[i][j] = harmful

        for d in deadly_areas:
            x1, y1, x2, y2 = [int(n) for n in d.split()]
            for i in range(x1, x2 + 1):
                for j in range(y1, y2 + 1):
                    matrix[i][j] = deadly 

        print str(matrix)
       
if __name__ == "__main__":
    esc = Escape()
    esc.lowest(["500 0 0 500"], ["0 0 0 0"])
