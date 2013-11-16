#!/usr/bin/env python

class Escape:
    
    def lowest(self, harmful_areas, deadly_areas):
        
        safe = 0
        harmful = 1
        deadly = 2
        size = 5 

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

        
if __name__ == "__main__":
    esc = Escape()
    esc.lowest(["5 0 0 5"], ["0 0 0 0"])
