import networkx as nx
import numpy as np
from .Neighborhood import Neighborhood

# get the weight of an edge
def getWeight(G,u,v):
    weight = G.get_edge_data(u,v,default = 1.0)
    if not isinstance(weight, (int, float, complex)): 
        #print(f"weight: {weight}")
        weight = weight['weight']
    return (weight)

def find_neighborhood(G, id, r):
    """
    Find the local neighborhood of a node considering primitive cycles up to size r.

    Notes
    -----
    - Only works for r <= 3.
    - Large values of r may require listing all paths up to length r-2, resulting in a complexity of O(d_{max}^r).

    Parameters
    ----------
    G : networkx.Graph
        Weighted undirected graph with self-loops.
    id : int
        Node identifier.

    Returns
    -------
    edges : list of tuples
        Edges that pertain to primitive cycles of length at most r.
    """
    n = G.number_of_nodes()
    list_edges = set()
    # initialize queue
    Q = []
    # initial node initialization
    father = {id: -1}
    visited = {id: True}
    level = {id:0}
    # always add all nodes at distance 1
    for neigh in G[id]:
        weight = getWeight(G,id,neigh)
        list_edges.add((id,neigh,weight))
        list_edges.add((neigh,id,weight))
        if id == neigh: # self loop add edge but do not include in the queue
            continue
        level[neigh] = level[id] + 1
        father[neigh] = id
        Q.append(neigh)
    # apply BFS
    while len(Q) != 0:
        u = Q.pop(0)
        if (level[u] << 1) > r + 2:
            break
        #
        visited[u] = True
        #
        for v in G[u]:
            if (v == father[u]) or (v == u): # visited to its parent or himself
                continue
            if v in visited.keys():
                if (father[v] == father[u]) and (r == 3): # have the same father (r = 3)
                    continue
                if(level[u] + level[v] + 1) <= r + 2:
                    weight = getWeight(G,u,v)
                    list_edges.add((u,v,weight))
                    list_edges.add((v,u,weight))
                    if father[v] != -1:
                        weight = getWeight(G,father[v],v)
                        list_edges.add((father[v],v,weight))
                        list_edges.add((v,father[v],weight))
                    if father[u] != -1:
                        weight = getWeight(G,u,father[u])
                        list_edges.add((father[u],u,weight))
                        list_edges.add((u,father[u],weight))
            elif v not in level.keys():
                level[v] = level[u] + 1
                father[v] = u
                Q.append(v)
    edges = []
    for edge in list_edges:
        edges.append(edge)
    #Graph_u.get_root()
    return edges

# Obtain the local neighborhood of N_{u\v} given the local neighborhoods of N_{u} and N_{v}
def neighborhood_difference(Graph_u,Graph_v):
    edges = list(Graph_u.get_edges().difference(Graph_v.get_edges()))
    direct_edges = [1 for u,v,_ in edges if (((u == Graph_u.get_root()) or (v == Graph_u.get_root())) and (u != v))] # only count the edges connected to the root, without the self-loop
    if len(direct_edges) == 0:
        edges = [(Graph_u.get_root(),Graph_u.get_root(),Graph_u.get_loop_weight())] # add the self-loop

    # nodes = set([u for u,_ in edges] + [v for _,v in edges])
    G_uv = Neighborhood(Graph_u.get_root(),edges)
    return G_uv
