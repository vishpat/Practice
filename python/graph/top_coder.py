#!/usr/bin/env python

from collections import defaultdict
from shortest_path import shortest_path
from shortest_path import shortest_distance


# SRM 150 RoboCourier
class RoboCourier:

    def get_instructions(self, path):
        instructions = ""

        mapping = {
            (0, 1)  : 'F',
            (1, 0)  : 'RF',
            (1, -1) : 'RRF',
            (0, -1) : 'RRRF',
            (-1, 0) : 'LLF',
            (-1, 1) : 'LF'

        }

        edges = len(path) - 1

        for i in range(0, edges):
            n1 = path[i]
            n2 = path[i + 1]

            diff_x = n2[0] - n1[0]
            diff_y = n2[1] - n1[1]
            
            move = mapping[(diff_x, diff_y)]
            
            instructions += move
        
        return instructions


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
        instructions = self.get_instructions(path)
        return path, instructions

if __name__ == "__main__":
    import sys

    rb = RoboCourier()
    path, instructions = rb.timeToDeliver(sys.argv[1])
    print str(path)
    print str(instructions)
