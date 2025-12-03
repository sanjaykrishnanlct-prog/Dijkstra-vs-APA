# Dijkstra vs A* Algorithm Performance Analysis

## Project Overview
This project implements and compares Dijkstra's algorithm and A* search algorithm for shortest path finding in weighted graphs. The implementation includes an optimized priority queue, admissible heuristics, and comprehensive performance testing.

## Algorithms Implemented

### 1. Dijkstra's Algorithm
- Finds shortest paths from source to all vertices
- Uses priority queue for O(E + V log V) complexity
- Guarantees optimality for non-negative weights

### 2. A* Search Algorithm
- Extends Dijkstra with heuristic guidance
- Uses admissible heuristics (Euclidean/Manhattan distance)
- Prunes search space while maintaining optimality

## Data Structures

### Priority Queue
- Binary heap implementation using Python's heapq
- O(log n) for push/pop operations
- Supports decrease-key operations
- Maintains uniqueness of elements

### Graph Representations
- Adjacency list for memory efficiency
- Node positions for heuristic calculations
- Support for directed/undirected weighted graphs

## Heuristic Functions

### Euclidean Distance
- Admissible and consistent for grid graphs
- Never overestimates actual distance
- Provides good guidance toward goal

### Manhattan Distance
- Admissible for grid movement (4-directional)
- Computationally cheaper than Euclidean
- Effective for grid-based pathfinding

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd dijkstra_astar_project

# Install dependencies
pip install -r requirements.txt
