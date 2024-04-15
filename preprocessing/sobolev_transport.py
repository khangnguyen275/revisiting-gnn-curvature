import functools

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx, to_networkx


class NXGraph():
    ''' Reconstruct PyG graphs as NetworkX graphs '''
    def __init__(self, graph, device = None):
        
        # if(device is None):
        #   self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # else:
        #   self.device = device
        
        self.graph = to_networkx(graph.data)
        self.edges = list(self.graph.edges)

    @functools.cache
    def get_dijkstra_path(self, u, v):
        return nx.single_source_dijkstra(self.graph, u, v)
        

    def sobolev_transport_1_hop(self, u, v):
        if (u,v) not in self.edges:
            raise LookupError(f'There is no edge between nodes {edge[0]} and {edge[1]}')

        # Define neighborhoods and support
        u_neighbors = list(self.graph.neighbors(u)) + [u]
        v_neighbors = list(self.graph.neighbors(v)) + [v]
        support = list(set(u_neighbors + v_neighbors))

        # Calculate shortest paths
        path_dict = {}
        edge_set = set()
        for neighbor in support:
            __, path_vertex = self.get_dijkstra_path(u, neighbor)
            path_edge = [(path_vertex[i], path_vertex[i+1]) for i in range(len(path_vertex)-1)]
            path_dict[neighbor] = path_edge
            edge_set.update(path_edge)
        edge_list = list(edge_set)

        # Define the computational matrix h
        h = np.zeros((len(support), len(edge_list)))
        for node in support:
            for edge in path_dict[node]: 
                h[support.index(node)][edge_list.index(edge)] = 1

        # Define the measure
        deg_u = 1/len(u_neighbors)
        deg_v = 1/len(v_neighbors)
        measure_u = np.zeros((len(support), 1))
        measure_v = np.zeros((len(support), 1))
        for node in support:
            if node in u_neighbors:
                measure_u[support.index(node)] = deg_u
            if node in v_neighbors:
                measure_v[support.index(node)] = deg_v

        H_u = np.matmul(h.T, measure_u)
        H_v = np.matmul(h.T, measure_v)

        return np.linalg.norm((H_u - H_v), ord=1)

        
        

    

    # def visualize(self):
    #     G = nx.Graph()
    #     G.add_edges_from(self.E)
    #     nx.draw_networkx(G)
    #     plt.show()