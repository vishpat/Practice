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
                cur_angle += 300
                cur_angle %= 360
                distance += 3
       
        end_state = cur_state

        return start_state, end_state, graph

    def opposite_inst(self, inst):
        
        if inst == 'L':
            return 'R'
        
        if inst == 'R':
            return 'L'

        if inst == 'LL':
            return 'RR'

        if inst == 'RR':
            return 'LL'    

    def get_min_moves(self, path):
        
        pos_x_dir = 1
        pos_y_dir = 2
        pos_z_dir = 3
        
        neg_x_dir = -pos_x_dir
        neg_y_dir = -pos_y_dir
        neg_z_dir = -pos_z_dir

        inst_mapping = {
            (pos_x_dir, pos_y_dir) : 'L', 
            (pos_x_dir, pos_z_dir) : 'R', 
            (pos_x_dir, neg_y_dir) : 'RR', 
            (pos_x_dir, neg_z_dir) : 'LL', 
            (pos_y_dir, pos_x_dir) : 'R', 
            (pos_y_dir, pos_z_dir) : 'RR', 
            (pos_y_dir, neg_x_dir) : 'LL', 
            (pos_y_dir, neg_z_dir) : 'L', 
            (pos_z_dir, pos_x_dir) : 'L', 
            (pos_z_dir, pos_y_dir) : 'LL', 
            (pos_z_dir, neg_x_dir) : 'RR', 
            (pos_z_dir, neg_y_dir) : 'R', 
        }
        
        dir_mapping = {
            (0, 1) : pos_y_dir, 
            (1, 0) : pos_x_dir,
           (1, -1) : pos_z_dir,
           (0, -1) : neg_y_dir,
           (-1, 0) : neg_x_dir,
           (-1, 1) : neg_z_dir
        }

        instructions = str() 

        edge_cnt = len(path) - 1
        priv_dir = pos_y_dir 

        for node_idx in xrange(0, edge_cnt):
            n1 = path[node_idx]
            n2 = path[node_idx + 1]

            dx = n2[0] - n1[0]
            dy = n2[1] - n1[1]
            
            cur_dir  = dir_mapping[(dx, dy)]  
            
            if cur_dir != priv_dir:
                if cur_dir == -1*priv_dir:
                    instructions += 'LLL'
                elif inst_mapping.has_key((priv_dir, cur_dir)):
                    instructions += inst_mapping[(priv_dir, cur_dir)]
                elif inst_mapping.has_key((-1*priv_dir, -1*cur_dir)):
                    instructions += inst_mapping[(-1*priv_dir, -1*cur_dir)]
                else:    
                    instructions += self.opposite_inst(inst_mapping[(cur_dir, priv_dir)])
                
            instructions += 'F'             
 
            priv_dir = cur_dir 

        return instructions

    def get_cost(self, instructions):
        cost = 0
        instruction_cnt = len(instructions)

        for i in range(0, instruction_cnt):
            instruction = instructions[i]

            if instruction == 'F':
                
                if (i > 0 and i < (instruction_cnt - 1) and 
                    instructions[i - 1] == 'F' and instructions[i + 1] == 'F'):
                    cost += 2
                else:    
                    cost += 4
            elif instruction == 'R' or instruction == 'L':
                cost += 3
    
        return cost


    def timeToDeliver(self, path_str):
        start, end, graph = self.create_graph(path_str)           

        path = shortest_path(graph, start, end)
        instructions = self.get_min_moves(path)
        cost = self.get_cost(instructions)

        return path, instructions, cost

tests = [ 
    ('FRRFLLFLLFRRFLF', 15),
    ('RFLLF', 17),
    ('FLFRRFRFRRFLLFRRF', 0),
    ('FFFFFFFFFRRFFFFFFRRFFFFFFLLFFFFFFLLFFFFFFRRFFFF', 44),
    ('RFLLFLFLFRFRRFFFRFFRFFRRFLFFRLRRFFLFFLFLLFRFLFLRFFRFFLFLFFRFFLLFLLFRFRFLRLFLRRFLRFLFFLFFFLFLFFRLFRLFLLFLFLRLRRFLFLFRLFRF', 24),

    ('LLFLFRLRRLRFFLRRRRFFFLRFFRRRLLFLFLLRLRFFLFRRFFFLFLRLFFRRLRLRRFFFLLLRFRLLRFFLFRLFRRFRRRFRLRLRLFFLLFLFFRFLRFRRLLLRFFRRRLRFLFRRFLFFRLFLFLFRLLLLFRLLRFLLLFFFLFRFRRFLLFFLLLFFRLLFLRRFRLFFFRRFFFLLRFFLRFRRRLLRFFFRRLLFLLRLFRRLRLLFFFLFLRFFRLRLLFLRLFFLLFFLLFFFRRLRFRRFLRRLRRLRFFFLLLLRRLRFFLFRFFRLLRFLFRRFLFLFFLFRRFRRLRRFLFFFLLRFLFRRFRFLRLRLLLLFLFFFLFRLLRFRLFRLFRLLFLFRLFFFFFFFRRLRLRLLRFLRLRRRRRRRRLFLFLFLRFLFRLFFRLFRRLLRRRRFFFRRRLLLLRRLFFLLLLLRFFFFRFRRLRRRFFFLLFFFFFLRRLRFLLRRLRLRFRRRRLFLLRFLRRFFFRFRLFFRLLFFRRLL', 169)
]

if __name__ == "__main__":
    for moves, cost in tests:
        rb = RoboCourier()
        path, instructions, min_cost = rb.timeToDeliver(moves)
        print str(instructions)
        assert cost == min_cost, "Expected %d got %d" % (cost, min_cost)
