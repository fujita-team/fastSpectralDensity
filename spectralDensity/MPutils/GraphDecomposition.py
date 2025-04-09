import networkx as nx
import numpy as np
from .find_neighborhood import find_neighborhood
from .find_neighborhood import neighborhood_difference
from .Neighborhood import Neighborhood

class GraphDecomposition:
    # Initializaes N_{node}
    def __init__(self, G, r = 0):
        """
        Class to save decomposition of the graph using primitive cycles of length at most r + 2
        
        Args:
            G: input graph
        """
        
        self.Graph = G
        self.r = r
        # recover all N_{i}
        self.nodes_neigh = dict()
        # recover all N_{i/j}
        self.cavity_neigh = dict() 
        # number of cavity nodes
        self.ncavity = 0
        # init node and edge neighboorhoods
        self.init_node_neighborhoods() # node neighborhood
        self.init_edge_neighborhoods() # cavity neighborhood
        ## 

    # Obtain the local neighborhood of N_{u} for all u
    def init_node_neighborhoods(self):
        for u in self.Graph.nodes:
            self.nodes_neigh[u] = Neighborhood(u, find_neighborhood(self.Graph,u,self.r))          
    
    #  Obtain the local neighborhood of N_{u\v} for all u,v
    def init_edge_neighborhoods(self):
        for u in self.Graph.nodes:
            self.cavity_neigh[u] = dict()
        # cavity[v][u] = G_{v \leftarrow u}
        self.ncavity = 0
        for u in self.Graph.nodes:
            for v in self.nodes_neigh[u].get_neighbors(): # v minus u
                self.cavity_neigh[v][u] = neighborhood_difference(self.nodes_neigh[v], self.nodes_neigh[u])
                self.ncavity = self.ncavity + 1
    # initialize all cavity neighbors values
    def initialize_cavity_messages(self):
        for u in self.Graph.nodes:
            for v in self.nodes_neigh[u].get_neighbors():
                self.cavity_neigh[v][u].init_var()

    def update_cavity_messages(self,z):
        """
        Update all cavity neighbors' variances.

        Parameters
        ----------
        z : complex
            Point where the spectral density is evaluated.

        Returns
        -------
        error : float
            Mean of the errors between the previous and new variance values.
        """
        error = 0
        for u in self.Graph.nodes:
            for v in self.nodes_neigh[u].get_neighbors():
                error += self.cavity_neigh[v][u].update_var(self.cavity_neigh,z)
        return error/self.ncavity        
    
    def compute_nodes_variance(self,z):
        """
        Compute node variances.

        Parameters
        ----------
        z : complex
            Point where the spectral density is evaluated.

        Returns
        -------
        variances : list of float
            List of all node variances obtained.
        """
        for u in self.Graph.nodes:
            self.nodes_neigh[u].update_var(self.cavity_neigh,z)
        all_var = [self.nodes_neigh[u].get_var() for u in self.Graph.nodes]
        return all_var
    
    # return neighbors of a node
    def get_neighborhood(self, i, j = None):
        if j is None:
            return self.nodes_neigh[i]
        else:
            return self.cavity_neigh[i][j]

    # Print the local neighborhood
    def print_decomp(self):
        for u in self.Graph.nodes:
            print("root is: ", u)
            print([(v, u) for v in self.nodes_neigh[u].get_neighbors()])
            for v in self.nodes_neigh[u].get_neighbors():
                print("neighbors of ",(v,u))
                if v != u:
                    print([(w,v) for w in self.cavity_neigh[v][u].get_neighbors()])
                    
