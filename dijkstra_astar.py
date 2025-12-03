import time
import math
from typing import Dict, List, Tuple, Optional, Callable
from .priority_queue import PriorityQueue

class GraphSearch:
    """
    Implementation of Dijkstra's algorithm and A* search algorithm.
    """
    
    @staticmethod
    def dijkstra(graph: Dict[int, List[Tuple[int, float]]], 
                 start: int, 
                 goal: int) -> Tuple[Optional[List[int]], float, int, float]:
        """
        Dijkstra's algorithm for finding shortest path.
        
        Args:
            graph: Adjacency list {node: [(neighbor, weight), ...]}
            start: Starting node
            goal: Goal node
            
        Returns:
            Tuple of (path, distance, nodes_expanded, execution_time)
        """
        start_time = time.time()
        
        # Initialize distances and predecessors
        distances = {node: float('inf') for node in graph}
        predecessors = {node: None for node in graph}
        distances[start] = 0
        
        # Priority queue
        pq = PriorityQueue()
        pq.push(start, 0)
        
        nodes_expanded = 0
        
        while not pq.is_empty():
            current = pq.pop()
            nodes_expanded += 1
            
            # Early termination if we reached the goal
            if current == goal:
                break
            
            current_dist = distances[current]
            
            # Explore neighbors
            for neighbor, weight in graph.get(current, []):
                new_dist = current_dist + weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current
                    pq.push(neighbor, new_dist)
        
        # Reconstruct path if exists
        if goal not in distances or distances[goal] == float('inf'):
            return None, float('inf'), nodes_expanded, time.time() - start_time
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        return path, distances[goal], nodes_expanded, time.time() - start_time
    
    @staticmethod
    def astar(graph: Dict[int, List[Tuple[int, float]]], 
              start: int, 
              goal: int,
              heuristic: Callable[[int, int], float],
              positions: Optional[List[Tuple[float, float]]] = None) -> Tuple[Optional[List[int]], float, int, float]:
        """
        A* search algorithm for finding shortest path with heuristic.
        
        Args:
            graph: Adjacency list {node: [(neighbor, weight), ...]}
            start: Starting node
            goal: Goal node
            heuristic: Heuristic function h(node, goal)
            positions: Optional node positions for Euclidean heuristic
            
        Returns:
            Tuple of (path, distance, nodes_expanded, execution_time)
        """
        start_time = time.time()
        
        # Initialize distances and predecessors
        g_score = {node: float('inf') for node in graph}
        f_score = {node: float('inf') for node in graph}
        predecessors = {node: None for node in graph}
        
        g_score[start] = 0
        f_score[start] = heuristic(start, goal) if positions is None else heuristic(start, goal, positions)
        
        # Priority queue
        pq = PriorityQueue()
        pq.push(start, f_score[start])
        
        nodes_expanded = 0
        
        while not pq.is_empty():
            current = pq.pop()
            nodes_expanded += 1
            
            # Early termination if we reached the goal
            if current == goal:
                break
            
            current_g = g_score[current]
            
            # Explore neighbors
            for neighbor, weight in graph.get(current, []):
                tentative_g = current_g + weight
                
                if tentative_g < g_score[neighbor]:
                    predecessors[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + (heuristic(neighbor, goal) if positions is None else heuristic(neighbor, goal, positions))
                    
                    if not pq.contains(neighbor):
                        pq.push(neighbor, f_score[neighbor])
        
        # Reconstruct path if exists
        if goal not in g_score or g_score[goal] == float('inf'):
            return None, float('inf'), nodes_expanded, time.time() - start_time
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        return path, g_score[goal], nodes_expanded, time.time() - start_time
    
    @staticmethod
    def euclidean_heuristic(node: int, goal: int, positions: List[Tuple[float, float]]) -> float:
        """
        Euclidean distance heuristic (admissible and consistent for grid graphs).
        
        Args:
            node: Current node
            goal: Goal node
            positions: List of (x, y) positions for each node
            
        Returns:
            Euclidean distance between node and goal
        """
        x1, y1 = positions[node]
        x2, y2 = positions[goal]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def manhattan_heuristic(node: int, goal: int, positions: List[Tuple[float, float]]) -> float:
        """
        Manhattan distance heuristic (admissible for grid graphs).
        
        Args:
            node: Current node
            goal: Goal node
            positions: List of (x, y) positions for each node
            
        Returns:
            Manhattan distance between node and goal
        """
        x1, y1 = positions[node]
        x2, y2 = positions[goal]
        return abs(x2 - x1) + abs(y2 - y1)
    
    @staticmethod
    def zero_heuristic(node: int, goal: int, positions: Optional[List[Tuple[float, float]]] = None) -> float:
        """
        Zero heuristic (turns A* into Dijkstra).
        
        Args:
            node: Current node
            goal: Goal node
            positions: Unused (for compatibility)
            
        Returns:
            Always 0
        """
        return 0.0