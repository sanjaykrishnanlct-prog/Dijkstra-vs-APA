import random
import math
from typing import List, Dict, Tuple, Set
import numpy as np

class GraphGenerator:
    """
    Generate random weighted graphs for testing Dijkstra and A* algorithms.
    """
    
    @staticmethod
    def generate_grid_graph(nodes: int, connect_prob: float = 0.3) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Tuple[float, float]]]:
        """
        Generate a grid-like graph with positions for Euclidean heuristic.
        
        Args:
            nodes: Number of nodes
            connect_prob: Probability of connecting two nearby nodes
            
        Returns:
            Tuple of (adjacency list, node positions)
        """
        # Arrange nodes in a grid
        side = int(math.sqrt(nodes))
        nodes = side * side  # Adjust to perfect square
        
        positions = []
        for i in range(side):
            for j in range(side):
                positions.append((float(i), float(j)))
        
        # Create adjacency list
        graph = {i: [] for i in range(nodes)}
        
        # Helper to convert grid coordinates to node index
        def coord_to_index(x, y):
            return x * side + y
        
        # Add edges
        for i in range(side):
            for j in range(side):
                current = coord_to_index(i, j)
                
                # Check right neighbor
                if j + 1 < side and random.random() < connect_prob:
                    neighbor = coord_to_index(i, j + 1)
                    weight = 1.0 + random.random() * 2.0  # Weight between 1 and 3
                    graph[current].append((neighbor, weight))
                    graph[neighbor].append((current, weight))
                
                # Check down neighbor
                if i + 1 < side and random.random() < connect_prob:
                    neighbor = coord_to_index(i + 1, j)
                    weight = 1.0 + random.random() * 2.0
                    graph[current].append((neighbor, weight))
                    graph[neighbor].append((current, weight))
                
                # Add some diagonal connections
                if i + 1 < side and j + 1 < side and random.random() < connect_prob / 2:
                    neighbor = coord_to_index(i + 1, j + 1)
                    weight = 1.414 + random.random() * 1.0  # Slightly longer for diagonals
                    graph[current].append((neighbor, weight))
                    graph[neighbor].append((current, weight))
        
        # Add some random long-range connections
        for _ in range(nodes // 10):
            u = random.randint(0, nodes - 1)
            v = random.randint(0, nodes - 1)
            if u != v and v not in [n for n, _ in graph[u]]:
                # Weight based on Euclidean distance
                dist = math.sqrt(
                    (positions[u][0] - positions[v][0])**2 +
                    (positions[u][1] - positions[v][1])**2
                )
                weight = dist * (0.8 + random.random() * 0.4)
                graph[u].append((v, weight))
                graph[v].append((u, weight))
        
        return graph, positions
    
    @staticmethod
    def generate_random_graph(nodes: int, edges_per_node: int = 5) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate a completely random graph.
        
        Args:
            nodes: Number of nodes
            edges_per_node: Average edges per node
            
        Returns:
            Adjacency list representation
        """
        graph = {i: [] for i in range(nodes)}
        
        total_edges = nodes * edges_per_node
        
        for _ in range(total_edges):
            u = random.randint(0, nodes - 1)
            v = random.randint(0, nodes - 1)
            if u != v:
                weight = 1.0 + random.random() * 9.0  # Weight between 1 and 10
                graph[u].append((v, weight))
        
        return graph
    
    @staticmethod
    def generate_layered_graph(nodes: int, layers: int = 10) -> Tuple[Dict[int, List[Tuple[int, float]]], List[Tuple[float, float]]]:
        """
        Generate a graph with clear layers (good for A*).
        
        Args:
            nodes: Number of nodes
            layers: Number of layers
            
        Returns:
            Tuple of (adjacency list, node positions)
        """
        nodes_per_layer = nodes // layers
        graph = {i: [] for i in range(nodes)}
        positions = []
        
        # Assign positions
        for layer in range(layers):
            layer_nodes = nodes_per_layer if layer < layers - 1 else nodes - layer * nodes_per_layer
            for i in range(layer_nodes):
                node_idx = layer * nodes_per_layer + i
                x = layer
                y = i - layer_nodes / 2
                positions.append((float(x), float(y)))
        
        # Connect layers
        for layer in range(layers - 1):
            current_layer_start = layer * nodes_per_layer
            next_layer_start = (layer + 1) * nodes_per_layer
            
            current_layer_nodes = nodes_per_layer
            next_layer_nodes = nodes_per_layer if layer + 1 < layers - 1 else nodes - (layer + 1) * nodes_per_layer
            
            # Connect each node to several nodes in next layer
            for i in range(current_layer_nodes):
                u = current_layer_start + i
                for _ in range(random.randint(1, 3)):
                    v = next_layer_start + random.randint(0, next_layer_nodes - 1)
                    weight = 1.0 + random.random() * 2.0
                    graph[u].append((v, weight))
        
        # Add some random connections within layers
        for layer in range(layers):
            layer_start = layer * nodes_per_layer
            layer_nodes = nodes_per_layer if layer < layers - 1 else nodes - layer * nodes_per_layer
            
            for i in range(layer_nodes):
                u = layer_start + i
                for _ in range(random.randint(0, 2)):
                    v = layer_start + random.randint(0, layer_nodes - 1)
                    if u != v:
                        weight = 0.5 + random.random() * 1.5
                        graph[u].append((v, weight))
        
        return graph, positions