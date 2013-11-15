#!/usr/bin/env python

from collections import defaultdict
from shortest_path import shortest_path
from shortest_path import shortest_distance


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

        start_state = cur_state = (0, 0)
        cur_angle = 0
        distance = 0
        
        instruction_cnt = len(path)

        for i in range(0, instruction_cnt):
            instruction = path[i]
            cur_x, cur_y = cur_state

            if instruction == 'F':
                
                if (i > 0 and i < (instruction_cnt - 1) and 
                    path[i - 1] == 'F' and path[i+ 1] == 'F'):
                    distance += 2
                else:    
                    distance += 4

                x, y = mapping[cur_angle]
                next_state = (cur_x + x, cur_y + y)

                neighbours = graph[cur_state]
                neighbours.append((next_state, distance))
                
                neighbours = graph[next_state]
                neighbours.append((cur_state, distance))
                
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
       
        end_state = cur_state

        return start_state, end_state, graph

    def timeToDeliver(self, path_str):
        start, end, graph = self.create_graph(path_str)           

        path = shortest_path(graph, start, end)
        shortest_dist = shortest_distance(graph, start, end)
  
        print str(path)
        return shortest_dist

if __name__ == "__main__":
    rb = RoboCourier()
    print str(rb.timeToDeliver("FFFFFFFFFRRFFFFFFRRFFFFFFLLFFFFFFLLFFFFFFRRFFFF"))
