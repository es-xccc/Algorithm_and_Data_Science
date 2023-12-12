# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:22:40 2022

@author: tiw
"""
def toChar(i):
    charstr='ABCDEFGHIJKLMNOPQRSTUVW'
    return charstr[i]

class Graph:
    def __init__(self, num_of_nodes):
        self.m_num_of_nodes = num_of_nodes
        self.m_graph = []

    def add_edge(self, node1, node2, weight):
        self.m_graph.append([node1, node2, weight])
        
    def find_subtree(self, parent, i):
        if parent[i] == i:
            return i
        return self.find_subtree(parent, parent[i])

# Connects subtrees containing nodes `x` and `y`
    def connect_subtrees(self, parent, subtree_sizes, x, y):
        xroot = self.find_subtree(parent, x)
        yroot = self.find_subtree(parent, y)
        if subtree_sizes[xroot] < subtree_sizes[yroot]:
            parent[xroot] = yroot
        elif subtree_sizes[xroot] > subtree_sizes[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            subtree_sizes[xroot] += 1

    def kruskals_mst(self):
        # Resulting tree
        result = []
        
        # Iterator
        i = 0
        # Number of edges in MST
        e = 0

        # Total weight of MST
        mst_weight = 0

        # Sort edges by their weight
        self.m_graph = sorted(self.m_graph, key=lambda item: item[2])
        
        # Auxiliary arrays
        parent = []
        subtree_sizes = []
        
        # Initialize `parent` and `subtree_sizes` arrays
        for node in range(self.m_num_of_nodes):
            parent.append(node)
            subtree_sizes.append(0)

        # Important property of MSTs
        # number of egdes in a MST is 
        # equal to (m_num_of_nodes - 1)
        while e < (self.m_num_of_nodes - 1):
            # Pick an edge with the minimal weight
            node1, node2, weight = self.m_graph[i]
            i = i + 1

            x = self.find_subtree(parent, node1)
            y = self.find_subtree(parent, node2)

            if x != y:
                e = e + 1
                result.append([node1, node2, weight])
                self.connect_subtrees(parent, subtree_sizes, x, y)
                mst_weight += weight
        
        # Print the resulting MST
        print("kruskals:")
        for node1, node2, weight in result:
            print("%s - %s: %d" % (toChar(node1), toChar(node2), weight))
        
        print("Weight of MST is %d" % mst_weight)
    
    def boruvkas_mst(self):
        parent = [i for i in range(self.m_num_of_nodes)]
        subtree_sizes = [0] * self.m_num_of_nodes
        cheapest = [-1] * self.m_num_of_nodes

        num_of_trees = self.m_num_of_nodes
        mst_weight = 0

        print("boruvkas:")
        while num_of_trees > 1:
            for i in range(len(self.m_graph)):
                u, v, w = self.m_graph[i]
                set1 = self.find_subtree(parent, u)
                set2 = self.find_subtree(parent, v)

                if set1 != set2:
                    if cheapest[set1] == -1 or cheapest[set1][2] > w:
                        cheapest[set1] = [u, v, w]

                    if cheapest[set2] == -1 or cheapest[set2][2] > w:
                        cheapest[set2] = [u, v, w]

            for node in range(self.m_num_of_nodes):
                if cheapest[node] != -1:
                    u, v, w = cheapest[node]
                    set1 = self.find_subtree(parent, u)
                    set2 = self.find_subtree(parent, v)

                    if set1 != set2:
                        mst_weight += w
                        self.connect_subtrees(parent, subtree_sizes, set1, set2)
                        print("%s - %s: %d" % (toChar(u), toChar(v), w))
                        num_of_trees = num_of_trees - 1

            cheapest = [-1] * self.m_num_of_nodes

        print("Weight of MST is %d" % mst_weight)


graph = Graph(9)
graph.add_edge(0, 1, 4)
graph.add_edge(0, 2, 7)
graph.add_edge(1, 2, 11)
graph.add_edge(1, 3, 9)
graph.add_edge(1, 5, 20)
graph.add_edge(2, 5, 1)
graph.add_edge(3, 6, 6)
graph.add_edge(3, 4, 2)
graph.add_edge(4, 6, 10)
graph.add_edge(4, 8, 15)
graph.add_edge(4, 7, 5)
graph.add_edge(4, 5, 1)
graph.add_edge(5, 7, 3)
graph.add_edge(6, 8, 5)
graph.add_edge(7, 8, 12)

graph.kruskals_mst()
print('-'*20)
graph.boruvkas_mst()
