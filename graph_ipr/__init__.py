"""
Graph-IPR: Learning LLM Agents via Dynamic State-Action Graph Refinement

This module implements the Graph-IPR framework that upgrades IPR from
linear trajectory sampling to global state graph optimization.
"""

from .state_graph import StateNode, ActionEdge, StateActionGraph
from .value_propagation import ValuePropagator
from .trajectory_stitching import TrajectoryStitcher, GraphPreferenceBuilder

__all__ = [
    'StateNode',
    'ActionEdge',
    'StateActionGraph',
    'ValuePropagator',
    'TrajectoryStitcher',
    'GraphPreferenceBuilder',
]
