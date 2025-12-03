import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dijkstra_astar import GraphSearch
from src.graph_generator import GraphGenerator
import csv
import json
import matplotlib.pyplot as plt
import numpy as np

def run_performance_tests():
    """Run performance tests on different graph sizes."""
    
    # Test configurations
    test_cases = [
        {"name": "Small Grid", "nodes": 1000, "type": "grid"},
        {"name": "Medium Grid", "nodes": 5000, "type": "grid"},
        {"name": "Large Grid", "nodes": 10000, "type": "grid"},
        {"name": "Small Layered", "nodes": 1000, "type": "layered"},
        {"name": "Medium Layered", "nodes": 5000, "type": "layered"},
        {"name": "Large Layered", "nodes": 10000, "type": "layered"},
    ]
    
    results = []
    
    print("=" * 80)
    print("DIJKSTRA vs A* PERFORMANCE COMPARISON")
    print("=" * 80)
    
    for test_case in test_cases:
        print(f"\n\nTesting {test_case['name']} ({test_case['nodes']} nodes)...")
        
        # Generate graph
        if test_case['type'] == 'grid':
            graph, positions = GraphGenerator.generate_grid_graph(test_case['nodes'])
        else:
            graph, positions = GraphGenerator.generate_layered_graph(test_case['nodes'])
        
        # Choose start and goal (opposite corners for grid, first and last for layered)
        # Use actual graph size, not requested size (grid graphs are adjusted to perfect squares)
        actual_nodes = len(graph)
        start = 0
        goal = actual_nodes - 1
        
        print(f"  Start node: {start}, Goal node: {goal}")
        print(f"  Graph has {len(graph)} nodes and {sum(len(neighbors) for neighbors in graph.values())} edges")
        
        # Run Dijkstra
        print("  Running Dijkstra...")
        dijkstra_path, dijkstra_dist, dijkstra_expanded, dijkstra_time = \
            GraphSearch.dijkstra(graph, start, goal)
        
        # Run A* with Euclidean heuristic
        print("  Running A* with Euclidean heuristic...")
        astar_path, astar_dist, astar_expanded, astar_time = \
            GraphSearch.astar(graph, start, goal, GraphSearch.euclidean_heuristic, positions)
        
        # Verify both algorithms found the same optimal distance
        if dijkstra_path is not None and astar_path is not None:
            path_match = abs(dijkstra_dist - astar_dist) < 1e-9
            if not path_match:
                print(f"  WARNING: Path distances differ! Dijkstra: {dijkstra_dist}, A*: {astar_dist}")
        
        # Calculate performance improvements
        expansion_reduction = ((dijkstra_expanded - astar_expanded) / dijkstra_expanded * 100) if dijkstra_expanded > 0 else 0
        time_reduction = ((dijkstra_time - astar_time) / dijkstra_time * 100) if dijkstra_time > 0 else 0
        
        # Store results
        result = {
            "Test Case": test_case['name'],
            "Nodes": test_case['nodes'],
            "Graph Type": test_case['type'],
            "Dijkstra Expansions": dijkstra_expanded,
            "A* Expansions": astar_expanded,
            "Expansion Reduction %": expansion_reduction,
            "Dijkstra Time (s)": dijkstra_time,
            "A* Time (s)": astar_time,
            "Time Reduction %": time_reduction,
            "Optimal Distance": dijkstra_dist if dijkstra_path else float('inf'),
            "Path Found": dijkstra_path is not None
        }
        
        results.append(result)
        
        # Print summary
        print(f"\n  Results for {test_case['name']}:")
        print(f"    Dijkstra: {dijkstra_expanded} nodes expanded, {dijkstra_time:.4f}s")
        print(f"    A*:       {astar_expanded} nodes expanded, {astar_time:.4f}s")
        print(f"    Reduction: {expansion_reduction:.1f}% fewer nodes, {time_reduction:.1f}% faster")
    
    # Save results to CSV
    with open('results/performance_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Test Case', 'Nodes', 'Graph Type', 'Dijkstra Expansions', 
                     'A* Expansions', 'Expansion Reduction %', 'Dijkstra Time (s)', 
                     'A* Time (s)', 'Time Reduction %', 'Optimal Distance', 'Path Found']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Test Case':<20} {'Nodes':<8} {'Dijkstra Exp':<12} {'A* Exp':<10} {'Exp Red %':<10} {'Dijkstra Time':<14} {'A* Time':<12} {'Time Red %':<10}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['Test Case']:<20} {result['Nodes']:<8} {result['Dijkstra Expansions']:<12} "
              f"{result['A* Expansions']:<10} {result['Expansion Reduction %']:<10.1f} "
              f"{result['Dijkstra Time (s)']:<14.4f} {result['A* Time (s)']:<12.4f} "
              f"{result['Time Reduction %']:<10.1f}")
    
    return results

def visualize_results(results):
    """Create visualizations of the performance results."""
    
    # Filter grid results for visualization
    grid_results = [r for r in results if r['Graph Type'] == 'grid']
    layered_results = [r for r in results if r['Graph Type'] == 'layered']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Node expansions comparison
    ax1 = axes[0, 0]
    x = np.arange(len(grid_results))
    width = 0.35
    
    dijkstra_exps = [r['Dijkstra Expansions'] for r in grid_results]
    astar_exps = [r['A* Expansions'] for r in grid_results]
    test_names = [r['Test Case'] for r in grid_results]
    
    ax1.bar(x - width/2, dijkstra_exps, width, label='Dijkstra', alpha=0.8)
    ax1.bar(x + width/2, astar_exps, width, label='A*', alpha=0.8)
    ax1.set_xlabel('Test Case')
    ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('Node Expansions Comparison (Grid Graphs)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Execution time comparison
    ax2 = axes[0, 1]
    dijkstra_times = [r['Dijkstra Time (s)'] for r in grid_results]
    astar_times = [r['A* Time (s)'] for r in grid_results]
    
    ax2.bar(x - width/2, dijkstra_times, width, label='Dijkstra', alpha=0.8)
    ax2.bar(x + width/2, astar_times, width, label='A*', alpha=0.8)
    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Execution Time Comparison (Grid Graphs)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Reduction percentages
    ax3 = axes[0, 2]
    exp_reductions = [r['Expansion Reduction %'] for r in grid_results]
    time_reductions = [r['Time Reduction %'] for r in grid_results]
    
    ax3.plot(x, exp_reductions, 'o-', label='Expansion Reduction', linewidth=2)
    ax3.plot(x, time_reductions, 's-', label='Time Reduction', linewidth=2)
    ax3.set_xlabel('Test Case')
    ax3.set_ylabel('Reduction (%)')
    ax3.set_title('Performance Improvement of A* over Dijkstra')
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Layered graphs - expansions
    ax4 = axes[1, 0]
    x_layered = np.arange(len(layered_results))
    
    dijkstra_exps_l = [r['Dijkstra Expansions'] for r in layered_results]
    astar_exps_l = [r['A* Expansions'] for r in layered_results]
    test_names_l = [r['Test Case'] for r in layered_results]
    
    ax4.bar(x_layered - width/2, dijkstra_exps_l, width, label='Dijkstra', alpha=0.8)
    ax4.bar(x_layered + width/2, astar_exps_l, width, label='A*', alpha=0.8)
    ax4.set_xlabel('Test Case')
    ax4.set_ylabel('Nodes Expanded')
    ax4.set_title('Node Expansions Comparison (Layered Graphs)')
    ax4.set_xticks(x_layered)
    ax4.set_xticklabels(test_names_l, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Layered graphs - times
    ax5 = axes[1, 1]
    dijkstra_times_l = [r['Dijkstra Time (s)'] for r in layered_results]
    astar_times_l = [r['A* Time (s)'] for r in layered_results]
    
    ax5.bar(x_layered - width/2, dijkstra_times_l, width, label='Dijkstra', alpha=0.8)
    ax5.bar(x_layered + width/2, astar_times_l, width, label='A*', alpha=0.8)
    ax5.set_xlabel('Test Case')
    ax5.set_ylabel('Execution Time (s)')
    ax5.set_title('Execution Time Comparison (Layered Graphs)')
    ax5.set_xticks(x_layered)
    ax5.set_xticklabels(test_names_l, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Scalability plot
    ax6 = axes[1, 2]
    all_nodes = [r['Nodes'] for r in results]
    all_dijkstra_times = [r['Dijkstra Time (s)'] for r in results]
    all_astar_times = [r['A* Time (s)'] for r in results]
    
    ax6.scatter(all_nodes, all_dijkstra_times, label='Dijkstra', alpha=0.6, s=100)
    ax6.scatter(all_nodes, all_astar_times, label='A*', alpha=0.6, s=100)
    
    # Add trend lines
    z_dijkstra = np.polyfit(all_nodes, all_dijkstra_times, 1)
    p_dijkstra = np.poly1d(z_dijkstra)
    z_astar = np.polyfit(all_nodes, all_astar_times, 1)
    p_astar = np.poly1d(z_astar)
    
    x_trend = np.linspace(min(all_nodes), max(all_nodes), 100)
    ax6.plot(x_trend, p_dijkstra(x_trend), '--', alpha=0.7)
    ax6.plot(x_trend, p_astar(x_trend), '--', alpha=0.7)
    
    ax6.set_xlabel('Number of Nodes')
    ax6.set_ylabel('Execution Time (s)')
    ax6.set_title('Scalability: Time vs Graph Size')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def heuristic_analysis():
    """Analyze different heuristic functions."""
    print("\n" + "=" * 80)
    print("HEURISTIC FUNCTION ANALYSIS")
    print("=" * 80)
    
    # Generate a test graph
    graph, positions = GraphGenerator.generate_grid_graph(2500)
    start = 0
    goal = 2499
    
    print(f"\nTesting different heuristics on 2500-node grid graph:")
    print(f"Start: {start}, Goal: {goal}")
    
    # Test different heuristics
    heuristics = [
        ("Zero (Dijkstra)", GraphSearch.zero_heuristic),
        ("Manhattan", GraphSearch.manhattan_heuristic),
        ("Euclidean", GraphSearch.euclidean_heuristic),
    ]
    
    results = []
    
    for name, heuristic_func in heuristics:
        print(f"\n  Testing {name} heuristic...")
        
        if name == "Zero (Dijkstra)":
            path, dist, expanded, time_taken = GraphSearch.dijkstra(graph, start, goal)
        else:
            path, dist, expanded, time_taken = GraphSearch.astar(
                graph, start, goal, heuristic_func, positions
            )
        
        results.append({
            "Heuristic": name,
            "Distance": dist,
            "Nodes Expanded": expanded,
            "Time (s)": time_taken,
            "Path Length": len(path) if path else 0
        })
        
        print(f"    Distance: {dist:.2f}")
        print(f"    Nodes expanded: {expanded}")
        print(f"    Time: {time_taken:.4f}s")
    
    # Print comparison
    print("\n" + "-" * 80)
    print(f"{'Heuristic':<20} {'Distance':<12} {'Nodes Expanded':<15} {'Time (s)':<10} {'Path Nodes':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['Heuristic']:<20} {result['Distance']:<12.2f} "
              f"{result['Nodes Expanded']:<15} {result['Time (s)']:<10.4f} "
              f"{result['Path Length']:<10}")

if __name__ == "__main__":
    print("Dijkstra vs A* Algorithm Performance Analysis")
    print("============================================\n")
    
    # Run performance tests
    results = run_performance_tests()
    
    # Run heuristic analysis
    heuristic_analysis()
    
    # Create visualizations (if matplotlib is available)
    try:
        visualize_results(results)
        print("\nVisualizations saved to 'results/performance_comparison.png'")
    except ImportError:
        print("\nNote: Install matplotlib for visualizations: pip install matplotlib")
    
    print("\n" + "=" * 80)
    print("Results have been saved to 'results/performance_results.csv'")
    print("=" * 80)
