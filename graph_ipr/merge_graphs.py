"""
Merge State Graphs from Parallel Workers

This script merges multiple state graphs produced by parallel workers
into a single unified graph.
"""

import os
import argparse
import glob
from pathlib import Path

from .state_graph import StateActionGraph


def merge_graphs(args):
    """Merge multiple sub-graphs into one"""

    # Find all graph files
    graph_files = glob.glob(os.path.join(args.input_dir, "**/state_graph.pkl"), recursive=True)

    if not graph_files:
        print(f"No graph files found in {args.input_dir}")
        return

    print(f"Found {len(graph_files)} graph files to merge")

    # Initialize merged graph
    merged_graph = StateActionGraph(
        task_name="merged",
        max_nodes=args.max_nodes,
        gamma=args.gamma,
        alpha=args.alpha,
    )

    # Merge each sub-graph
    for i, graph_file in enumerate(graph_files):
        try:
            sub_graph = StateActionGraph.load(graph_file)
            print(f"  [{i+1}/{len(graph_files)}] Merging {graph_file}: {len(sub_graph.nodes)} nodes")
            merged_graph.merge_from(sub_graph)
        except Exception as e:
            print(f"  Warning: Failed to load {graph_file}: {e}")
            continue

    print(f"\nMerged graph statistics:")
    stats = merged_graph.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save merged graph
    output_path = args.output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_graph.save(output_path)
    print(f"\nSaved merged graph to {output_path}")

    return merged_graph


def main():
    parser = argparse.ArgumentParser("Merge State Graphs from Parallel Workers")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing sub-graphs from parallel workers"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged graph"
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=10000,
        help="Maximum nodes in merged graph"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate"
    )

    args = parser.parse_args()
    merge_graphs(args)


if __name__ == "__main__":
    main()
