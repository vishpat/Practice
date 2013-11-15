#!/usr/bin/env python

from collections import defaultdict
import sys
import heapq

graph1 = {
   'O': [('A', 2), ('B', 5), ('C', 4)],
   'A': [('O', 2), ('B', 2), ('D', 7), ('F', 12)],
   'B': [('O', 5), ('C', 1), ('D', 4), ('E', 3), ('A', 2)],
   'C': [('O', 4), ('B', 1), ('E', 4)],
   'D': [('A', 7), ('B', 4), ('E', 1), ('T', 5)],
   'E': [('B', 3), ('C', 4), ('D', 1), ('T', 7)],
   'F': [('A', 12), ('T', 3)],
   'T': [('D', 5), ('E', 7), ('F', 3)]
}

graph2 = {
    'A' : [('B', 3), ('C', 2), ('D', 5)],
    'B' : [('A', 3), ('D', 2), ('F', 13)],
    'C' : [('A', 2), ('D', 2), ('E', 5) ],
    'D' : [('A', 5), ('B', 2), ('C', 2), ('E', 4), ('F', 6), ('G', 3)],
    'E' : [('C', 5), ('D', 4), ('G', 6)],
    'F' : [('B', 13), ('D', 6), ('G', 2), ('H', 3)],
    'G' : [('D', 3), ('E', 6), ('F', 2), ('H', 6)],
    'H' : [('F', 3), ('G', 6)]

}

def shortest_path(graph, current, end, parent, visited, distance):
 
    if current == end:
        return 
 
    neighbours = graph[current]
    for neighbour, neighbour_dist in neighbours:
        
        if visited.has_key(neighbour):
            continue

        if distance[current] + neighbour_dist < distance[neighbour]:
            distance[neighbour] = distance[current] + neighbour_dist 
            parent[neighbour] = current

    visited[current] = True

    h = list()
    for node in graph.keys():
        if not visited.has_key(node):
            heapq.heappush(h, (distance[node], node))
    
    _, min_neighbour = heapq.heappop(h)
    shortest_path(graph, min_neighbour, end, parent, visited, distance)      


def find_shortest(graph, start, end):
    visited = defaultdict(bool) 
    distance = {}
    parent = {} 

    for node in graph.keys():
        distance[node] = sys.maxint

    distance[start] = 0
    parent[start] = None

    shortest_path(graph, start, end, parent, visited, distance)
    
    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(cur)
    path.reverse()
    return path

if __name__ == "__main__":
    print str(find_shortest(graph2, 'A', 'H'))
