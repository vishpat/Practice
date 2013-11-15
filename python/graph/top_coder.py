#!/usr/bin/env python

from collections import defaultdict

# SRM 150 RoboCourier
class RoboCourier:

    def create_graph(self, path_str):
        
        graph = defaultdict(list) 
        
        mapping = {
             0  : (0, 1), 
            60  : (1, 0),
           120  : (1, -1),
           180  : (0, -1),
           240  : (-1, 0),
           300  : (-1, 1),
        }

        path = list(path_str)

        cur_state = (0, 0)
        cur_angle = 0
        distance = 0

        for instruction in path:
            cur_x, cur_y = cur_state

            if instruction == 'F':
                distance += 4
                x, y = mapping[cur_angle]
                next_state = (cur_x + x, cur_y + y)
                neighbours = graph[cur_state]
                neighbours.append((next_state, distance))
                cur_state = next_state
                distance = 0
            elif instruction == 'R':
                cur_angle += 60
                cur_angle %= 360
                distance += 3
            elif instruction == 'L':
                cur_angle -= 60
                cur_angle %= 360
                distance += 3

        return graph

    def timeToDeliver(self, path_str):
        graph = self.create_graph(path_str)            
        print str(graph)        

if __name__ == "__main__":
    rb = RoboCourier()
    rb.timeToDeliver("FRRFLLFLLFRRFLF")
