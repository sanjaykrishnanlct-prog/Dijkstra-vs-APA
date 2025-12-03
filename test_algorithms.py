import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.dijkstra_astar import GraphSearch
from src.graph_generator import GraphGenerator

class TestGraphAlgorithms(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test graph
        self.simple_graph = {
            0: [(1, 1.0), (2, 4.0)],
            1: [(2, 2.0), (3, 5.0)],
            2: [(3, 1.0)],
            3: []
        }
        
        # Positions for simple graph (grid-like)
        self.simple_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    def test_dijkstra_simple(self):
        """Test Dijkstra on a simple graph."""
        path, dist, expanded, time_taken = GraphSearch.dijkstra(
            self.simple_graph, 0, 3
        )
        
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 4)  # 0 -> 1 -> 2 -> 3
        self.assertAlmostEqual(dist, 4.0)  # 1 + 2 + 1 = 4
        self.assertGreater(expanded, 0)
    
    def test_astar_simple(self):
        """Test A* on a simple graph."""
        path, dist, expanded, time_taken = GraphSearch.astar(
            self.simple_graph, 0, 3,
            GraphSearch.euclidean_heuristic,
            self.simple_positions
        )
        
        self.assertIsNotNone(path)
        self.assertAlmostEqual(dist, 4.0)
        self.assertGreater(expanded, 0)
    
    def test_dijkstra_vs_astar_same_result(self):
        """Test that Dijkstra and A* find the same optimal path."""
        # Generate a random graph
        graph, positions = GraphGenerator.generate_grid_graph(100)
        
        # Test multiple start-goal pairs
        test_pairs = [(0, 99), (10, 80), (25, 75)]
        
        for start, goal in test_pairs:
            # Run Dijkstra
            dijkstra_result = GraphSearch.dijkstra(graph, start, goal)
            dijkstra_path, dijkstra_dist = dijkstra_result[0], dijkstra_result[1]
            
            # Run A*
            astar_result = GraphSearch.astar(
                graph, start, goal,
                GraphSearch.euclidean_heuristic,
                positions
            )
            astar_path, astar_dist = astar_result[0], astar_result[1]
            
            # Both should either find a path or not
            if dijkstra_path is None:
                self.assertIsNone(astar_path)
            else:
                self.assertIsNotNone(astar_path)
                # Distances should be equal (optimal)
                self.assertAlmostEqual(dijkstra_dist, astar_dist, places=6)
    
    def test_heuristic_admissibility(self):
        """Test that Euclidean heuristic is admissible."""
        graph, positions = GraphGenerator.generate_grid_graph(50)
        
        for start in range(0, 50, 10):
            for goal in range(0, 50, 10):
                if start != goal:
                    # Actual shortest path distance
                    dijkstra_result = GraphSearch.dijkstra(graph, start, goal)
                    actual_dist = dijkstra_result[1]
                    
                    # Heuristic estimate
                    heuristic_est = GraphSearch.euclidean_heuristic(
                        start, goal, positions
                    )
                    
                    # Heuristic must not overestimate (admissibility)
                    self.assertLessEqual(heuristic_est, actual_dist + 1e-9)
    
    def test_no_path(self):
        """Test when no path exists."""
        # Create disconnected graph
        disconnected_graph = {
            0: [(1, 1.0)],
            1: [],
            2: [(3, 1.0)],
            3: []
        }
        
        # No path from 0 to 2
        dijkstra_result = GraphSearch.dijkstra(disconnected_graph, 0, 2)
        self.assertIsNone(dijkstra_result[0])
        self.assertEqual(dijkstra_result[1], float('inf'))
        
        astar_result = GraphSearch.astar(
            disconnected_graph, 0, 2,
            GraphSearch.euclidean_heuristic,
            [(0,0), (1,0), (0,1), (1,1)]
        )
        self.assertIsNone(astar_result[0])
        self.assertEqual(astar_result[1], float('inf'))
    
    def test_same_start_goal(self):
        """Test when start and goal are the same node."""
        path, dist, expanded, time_taken = GraphSearch.dijkstra(
            self.simple_graph, 0, 0
        )
        
        self.assertEqual(path, [0])
        self.assertEqual(dist, 0.0)
        self.assertEqual(expanded, 1)

if __name__ == '__main__':
    unittest.main()