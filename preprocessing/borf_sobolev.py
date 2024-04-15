import os
import ot
import time
import torch
import pathlib
import numpy as np
import pandas as pd
import multiprocessing as mp
import networkx as nx
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
import random
import functools

class CalculationGraph(nx.Graph):
    '''Graph class inherited from the NetworkX Graph class with extra methods for calculating curvature'''

    def visualize_edge(self, edge):
        # Get the local neighborhood
        u, v = edge
        local_neighbors = set()
        local_neighbors = local_neighbors | set(self.neighbors(u)) | set(self.neighbors(v))

        # Add points that form a pentagon with regard to the edge
        pent_nodes = set()
        seen_nodes = local_neighbors.copy()
        for p in self.neighbors(u):
            for q in self.neighbors(p):
                if q in seen_nodes: 
                    continue
                seen_nodes.add(q)
                flag = False
                for w in self.neighbors(v):
                    if (q,w) in self.edges and w != p:
                        flag = True
                        continue
                if flag:
                    pent_nodes.add(q)
        
        # Draw the induced subgraph
        local_neighborhood = local_neighbors | pent_nodes
        labels = {}
        for node in local_neighborhood:
            labels[node] = ''
        labels[u] = u
        labels[v] = v
        local_neighborhood = self.subgraph(local_neighborhood)        
        nx.draw_networkx(local_neighborhood, labels = labels)
    
    @functools.cache
    def _get_dijkstra_path(self, u, v):
        return nx.single_source_dijkstra(self, u, v)

    def _sobolev_transport_1_hop(self, u, v):
        if (u,v) not in self.edges:
            raise LookupError(f'There is no edge between nodes {edge[0]} and {edge[1]}')

        u_neighbors = list(self.neighbors(u)) + [u]
        v_neighbors = list(self.neighbors(v)) + [v]
        support = list(set(u_neighbors + v_neighbors))

        # Calculate shortest paths
        path_dict = {}
        edge_set = set()
        for neighbor in support:
            __, path_vertex = self._get_dijkstra_path(u, neighbor)
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
        neighbor_mass_u = 1/len(u_neighbors)
        neighbor_mass_v = 1/len(v_neighbors)
        measure_u = np.zeros(len(support))
        measure_v = np.zeros(len(support))
        for node in support:
            if node in u_neighbors:
                measure_u[support.index(node)] = neighbor_mass_u
            if node in v_neighbors:
                measure_v[support.index(node)] = neighbor_mass_v

        H_u = np.matmul(h.T, measure_u)
        H_v = np.matmul(h.T, measure_v)

        return np.linalg.norm((H_u - H_v), ord=1)

    @functools.cache
    def _wasserstein_transport_1_hop(self, u, v):
        # Get general neighborhood informations
        u_neighbors = list(self.neighbors(u)) + [u]
        v_neighbors = list(self.neighbors(v)) + [v]
        deg_u = len(u_neighbors)
        deg_v = len(v_neighbors)

        # Define the measure
        neighbor_mass_u = 1/deg_u
        neighbor_mass_v = 1/deg_v
        measure_u = np.ones(deg_u) * 1/deg_u
        measure_v = np.ones(deg_v) * 1/deg_v

        # Define the distance matrix
        distance_matrix = np.full((deg_u, deg_v), np.inf)
        for node_1 in u_neighbors:
            for node_2 in v_neighbors:
                index = (u_neighbors.index(node_1), v_neighbors.index(node_2))
                if distance_matrix[index] == np.inf:
                    distance_matrix[index], __ = self._get_dijkstra_path(node_1, node_2)
        return ot.emd2(measure_u, measure_v, distance_matrix)

    def compute_ricci_curvature(self, transport_type = 'sobolev_onlyu'):
        ''' 
        supported curvature types: wasserstein, sobolev_onlyu 
        '''
        if transport_type == 'wasserstein':
            transport_dist = self._wasserstein_transport_1_hop
        elif transport_type == 'sobolev_onlyu':
            transport_dist = self._sobolev_transport_1_hop

        curvature_dict = {}
        for edge in self.edges:
            curvature_dict[edge] = 1 - transport_dist(*edge)
            
        self._get_dijkstra_path.cache_clear()
        # transport_dist.cache_clear()
        return curvature_dict

# def convert_calculation_graph(pyg_graph):
#     graph = to_networkx(pyg_graph.data)
#     return CalculationGraph(graph)

def borf(
    data,
    loops = 5,
    remove_edges = True,
    removal_bound = 0.5,
    is_undirected = True,
    batch_add = 20,
    batch_remove = 10,
    device = None,
    save_dir = 'rewired_graphs',
    transport_type = 'sobolev_onlyu',
    dataset_name = None,
    graph_index = 0,
    debug = False
):
        # Check if there is a preprocessed graph
    dirname = f'{save_dir}/{transport_type}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(dirname, f'loop_{loops}_add_{batch_add}_remove_{batch_remove}_edge_index_{graph_index}.pt')
    edge_type_filename = os.path.join(dirname, f'loop_{loops}_add_{batch_add}_remove_{batch_remove}_edge_type_{graph_index}.pt')

    if(os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename)):
        if(debug) : print(f'[INFO] Rewired graph for {loops} batches, {batch_add} edge additions and {batch_remove} edge removal exists...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        return edge_index, edge_type

    # Preprocess data
    # G, N, edge_type = _preprocess_data(data)

    # Rewiring begins
    for _ in range(loops):
        # Compute ORC
        G = CalculationGraph(to_networkx(data))
        # largest_cc = max(nx.connected_components(G), key=len)
        curvature = G.compute_ricci_curvature(transport_type = 'sobolev_onlyu')
        _C = sorted(G.edges, key=lambda edge: curvature[edge])

        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]

        # Add edges
        for (u, v) in most_neg_edges:
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
            u_neighbor = random.choice(list(u_neighbors-v_neighbors))
            v_neighbor = random.choice(list(v_neighbors-u_neighbors))
            G.add_edge(u_neighbor, v_neighbor)

        # Remove edges
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)

    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)
    # edge_type = torch.tensor(edge_type)

    if(debug) : print(f'[INFO] Saving edge_index to {edge_index_filename}')
    with open(edge_index_filename, 'wb') as f:
        torch.save(edge_index, f)

    if(debug) : print(f'[INFO] Saving edge_type to {edge_type_filename}')
    with open(edge_type_filename, 'wb') as f:
        torch.save(edge_type, f)

    return edge_index, edge_type
    
