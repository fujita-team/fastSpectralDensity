import numpy as np

class Neighborhood:
    # Initializaes N_{node}
    def __init__(self, node, edges):
        """
        Class to save the local neighborhoof of "node" given the list of edges
        
        Args:
            node: node to which the local neighborhood pertains
            edges: edges corresponding to the local neighborhood
        
        Returns:
            None
        """
        
        self.root = node
        self.loop_weight = 0
        self.edges = set(edges) # (u,v,weight)
        # separate edges, edges directly connected to the node and edges that are not connected among them
        # also we create a set with the nodes in the neighborhood
        self.node_idx = dict() # index of each node in the neighborhood (some order)
        self.neigh_nodes = None
        self.A = None
        self.V = None
        self.nnodes = None
        self.var = 1+1.j # variance of the complex Gaussian distribution
        self.error = None
        # Initialize the matrices/vectors and self loop weight
        self.init_neighborhood()
    #
    def print_neigh(self):
        print(self.root)
        print(self.edges)
    # initialization of the current node
    def init_neighborhood(self):
        # nodes in the local neighborhood
        self.neigh_nodes = set()
        for u,v,_ in self.edges:
            self.neigh_nodes.add(u)
            self.neigh_nodes.add(v)
        # remove node to itself
        self.neigh_nodes.remove(self.root)
        # creating indices
        self.node_idx = {node:idx for idx,node in enumerate(self.neigh_nodes)}
        self.nnodes = len(self.neigh_nodes)
        #
        self.A = np.zeros((self.nnodes,self.nnodes),dtype=np.complex128)
        self.V = np.zeros((self.nnodes,1),dtype=np.complex128)
        for u,v,w in self.edges:
            if u == self.root and v == self.root:
                self.loop_weight = w
            elif (u == self.root):
                self.V[self.node_idx[v]] = w
            elif (v == self.root):
                self.V[self.node_idx[u]] = w
            else:
                i,j = self.node_idx[u],self.node_idx[v]
                self.A[i,j] = self.A[j,i] = w 
    # function that returns the number of nodes in the local neighborhood
    # without including the root
    def get_num_nodes(self):
        return self.nnodes

    def update_var(self,cavityNeigh,z):
        """
        Update the internal values of the node, given the other values.

        Parameters
        ----------
        cavityNeigh : list
            Cavity neighborhood.
        z : complex
            Complex number, required to obtain the spectral density.

        Returns
        -------
        error : float
            The error between the previous and actual variance value.
        """
        if len(self.node_idx) == 0:
            self.error = 0
            self.var = self.loop_weight
        # building the diagonal matrix
        diag = [(z - cavityNeigh[neigh][self.root].get_var()) for neigh in self.node_idx]
        D = np.diag(diag)
        S = D - self.A
        # now solve the recurrence to updatethe value
        new_val = np.dot(self.V.T,np.linalg.solve(S,self.V)) + self.loop_weight
        self.error = np.abs(new_val[0,0] - self.var)
        self.var = new_val[0,0]
        return (self.error)

    # return the root of the decomposition
    def get_root(self):
        return self.root
    # return self loop weight
    def get_loop_weight(self):
        return self.loop_weight

    # return the variance
    def get_var(self):
        return self.var
    # return the edge set
    def get_edges(self):
        return self.edges
    
    # return the nodes of this local neighborhood
    def get_nodes(self):
        return self.Graph.nodes
    
    # return the neighbors of a vertex, considering the local neighborhood
    def get_neighbors(self):
        return self.neigh_nodes
    
    # return the neighbors of a vertex that are directly connected to the node
    def get_direct_neighbors(self):
        return self.direct_neigh_nodes
    
    # return the adjacency matrix
    def get_adj(self):
        return self.A
    # initialize variance and error
    def init_var(self):
        self.var = 1+1.j
        self.error = None
    # return the edge set
    def get_edges(self):
        return self.edges
    # return the nodes of this local neighborhood
    def get_nodes(self):
        return self.Graph.nodes

    # return the neighbors of a vertex, considering the local neighborhood
    def get_neighbors(self):
        return self.neigh_nodes

    # return the adjacency matrix
    def get_adj(self):
        return self.A
    # return the vector of inmediate neighbors
    def get_v(self):
        return self.V
