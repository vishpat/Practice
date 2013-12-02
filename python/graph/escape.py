#!/usr/bin/env python

import sys
from shortest_path import shortest_path
from shortest_path import shortest_distance


safe = 0
harmful = 1
deadly = sys.maxint 
size = 500 + 1 

# TCI '02 Semifinals 2 (Division I Level Two)

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
            for i in range(min(x1, x2), max(x1, x2) + 1):
                for j in range(min(y1, y2), max(y1, y2) + 1):
                    matrix[i][j] = harmful

        for d in deadly_areas:
            x1, y1, x2, y2 = [int(n) for n in d.split()]
            for i in range(min(x1, x2), max(x1, x2) + 1):
                for j in range(min(y1, y2), max(y1, y2) + 1):
                    matrix[i][j] = deadly 

        graph = {}
        for i in range(0, size):
            for j in range(0, size):

                state = matrix[i][j]
                
                if (state == deadly and not (i == 0 and j == 0)):
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

        matrix = None 

        min_path = shortest_path(graph, (0, 0), (size - 1, size - 1))
        min_dist = shortest_distance(graph, (0, 0), (size - 1, size - 1))

        return min_dist

    def run_tests(self):
        tests = [ 
            ([], [], 0),
            (["500 0 0 500"], ["0 0 0 0"], 1000),
            (["0 0 250 250", "250 250 500 500"], ["0 251 249 500","251 0 500 249"], 1000),
            (["0 0 250 250", "250 250 500 500"], ["0 250 250 500", "250 0 500 250"], -1),
            ([
                "468 209 456 32",
                "71 260 306 427",
                "420 90 424 492",
                "374 253 54 253",
                "319 334 152 431",
                "38 93 204 84",
                "246 0 434 263",
                "12 18 118 461",
                "215 462 44 317",
                "447 214 28 475",
                "3 89 38 125",
                "157 108 138 264",
                "363 17 333 387",
                "457 362 396 324",
                "95 27 374 175",
                "381 196 265 302",
                "105 255 253 134",
                "0 308 453 55",
                "169 28 313 498",
                "103 247 165 376",
                "264 287 363 407",
                "185 255 110 415",
                "475 126 293 112",
                "285 200 66 484",
                "60 178 461 301",
                "347 352 470 479",
                "433 130 383 370",
                "405 378 117 377",
                "403 324 369 133",
                "12 63 174 309",
                "181 0 356 56",
                "473 380 315 378"],
            [
                "250 384 355 234",
                "28 155 470 4",
                "333 405 12 456",
                "329 221 239 215",
                "334 20 429 338",
                "85 42 188 388",
                "219 187 12 111",
                "467 453 358 133",
                "472 172 257 288",
                "412 246 431 86",
                "335 22 448 47",
                "150 14 149 11",
                "224 136 466 328",
                "369 209 184 262",
                "274 488 425 195",
                "55 82 279 253",
                "153 201 65 228",
                "208 230 132 223",
                "369 305 397 267",
                "200 145 98 198",
                "422 67 252 479",
                "231 252 401 190",
                "312 20 0 350",
                "406 72 207 294",
                "488 329 338 326",
                "117 264 497 447",
                "491 341 139 438",
                "40 413 329 290",
                "148 245 53 386",
                "147 70 186 131",
                "300 407 71 183",
                "300 186 251 198",
                "178 67 487 77",
                "98 158 55 433",
                "167 231 253 90",
                "268 406 81 271",
                "312 161 387 153",
                "33 442 25 412",
                "56 69 177 428",
                "5 92 61 247"
                ],
                254
            )
        ]

        for harmful, deadly, cost in tests:
            min_cos = self.lowest(harmful, deadly)
            assert min_cos == cost, "Did not find the expected lowest cost min_cost %d  expected %d" % (min_cos, cost)

if __name__ == "__main__":
    esc = Escape()
    esc.run_tests()
