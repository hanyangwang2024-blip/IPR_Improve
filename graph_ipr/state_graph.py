"""
State-Action Graph: Core data structure for Graph-IPR

This module implements the state graph that accumulates knowledge from
all exploration trajectories across iterations.
"""

import hashlib
import json
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from pathlib import Path


@dataclass
class StateNode:
    """State node in the graph"""
    state_id: str                           # Unique identifier (MD5 hash)
    task_id: int                            # Task ID (separates state spaces)
    step: int                               # Step number in trajectory
    observation: str                        # Environment observation (last user message)
    history_summary: str                    # Action sequence summary
    is_terminal: bool = False               # Whether this is a terminal state
    terminal_reward: Optional[float] = None # Terminal reward (only valid for terminal states)

    # Visit statistics
    visit_count: int = 0                    # Number of visits
    last_visit_time: int = 0                # Last visit timestamp (for LRU)

    def __hash__(self):
        return hash(self.state_id)

    def __eq__(self, other):
        if not isinstance(other, StateNode):
            return False
        return self.state_id == other.state_id

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'state_id': self.state_id,
            'task_id': self.task_id,
            'step': self.step,
            'observation': self.observation[:500],  # Truncate for storage
            'history_summary': self.history_summary,
            'is_terminal': self.is_terminal,
            'terminal_reward': self.terminal_reward,
            'visit_count': self.visit_count,
            'last_visit_time': self.last_visit_time,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StateNode':
        """Deserialize from dictionary"""
        return cls(
            state_id=data['state_id'],
            task_id=data['task_id'],
            step=data['step'],
            observation=data['observation'],
            history_summary=data['history_summary'],
            is_terminal=data['is_terminal'],
            terminal_reward=data.get('terminal_reward'),
            visit_count=data.get('visit_count', 0),
            last_visit_time=data.get('last_visit_time', 0),
        )


@dataclass
class ActionEdge:
    """Action edge connecting two states"""
    source_id: str                  # Source state ID
    action: str                     # Action content (agent's full output)
    target_id: str                  # Target state ID

    # Transition statistics
    immediate_reward: float = 0.0   # Immediate reward r(s, a)
    transition_count: int = 0       # Transition count n(s, a, s')

    # Q-value (updated by TD)
    q_value: float = 0.0

    def confidence(self) -> float:
        """Confidence score: more exploration means more confidence"""
        return np.sqrt(self.transition_count)

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'source_id': self.source_id,
            'action': self.action[:1000],  # Truncate for storage
            'target_id': self.target_id,
            'immediate_reward': self.immediate_reward,
            'transition_count': self.transition_count,
            'q_value': self.q_value,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ActionEdge':
        """Deserialize from dictionary"""
        return cls(
            source_id=data['source_id'],
            action=data['action'],
            target_id=data['target_id'],
            immediate_reward=data.get('immediate_reward', 0.0),
            transition_count=data.get('transition_count', 0),
            q_value=data.get('q_value', 0.0),
        )


class StateActionGraph:
    """
    State-Action Graph: Core data structure for Graph-IPR

    Accumulates (s, a, s', r) transitions from all trajectories and
    provides Q-value estimation through value propagation.
    """

    def __init__(
        self,
        task_name: str = "webshop",
        max_nodes: int = 10000,
        gamma: float = 0.95,
        alpha: float = 0.1,
    ):
        """
        Initialize the state-action graph.

        Args:
            task_name: Name of the task (webshop, alfworld, etc.)
            max_nodes: Maximum number of nodes before pruning
            gamma: Discount factor for value propagation
            alpha: Learning rate for Q-value updates
        """
        self.task_name = task_name
        self.max_nodes = max_nodes
        self.gamma = gamma
        self.alpha = alpha

        # Core storage
        self.nodes: Dict[str, StateNode] = {}
        # edges[source_id][action_hash] -> List[ActionEdge]
        self.edges: Dict[str, Dict[str, List[ActionEdge]]] = defaultdict(lambda: defaultdict(list))
        # 反向索引: incoming_edges[target_id] -> Set[source_id]，用于快速删除节点
        self.incoming_edges: Dict[str, Set[str]] = defaultdict(set)

        # Index structures
        self.task_states: Dict[int, Set[str]] = defaultdict(set)  # task_id -> {state_ids}
        self.terminal_states: Set[str] = set()                     # Terminal state set
        self.successful_paths: List[List[str]] = []                # Successful trajectory state sequences

        # Timestamp for LRU
        self.current_time: int = 0

        # Statistics
        self.total_trajectories: int = 0
        self.successful_trajectories: int = 0

    def _compute_state_id(self, task_id: int, observation: str, history_summary: str) -> str:
        """
        Compute unique state identifier using MD5 hash.

        The state is identified by:
        - task_id: Different tasks have different state spaces
        - history_summary: Sequence of recent actions
        - observation: Current environment observation
        """
        content = f"{task_id}||{history_summary}||{observation[:200]}"
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_history_summary(self, history: List[Dict]) -> str:
        """
        Extract action sequence summary from conversation history.

        Only keeps the last 5 actions to balance between state distinction
        and generalization.
        """
        actions = []
        for msg in history:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Extract Action part
                if 'Action:' in content:
                    action_part = content.split('Action:')[-1].strip()
                    # Take first line of action
                    action_line = action_part.split('\n')[0].strip()
                    actions.append(action_line)

        # Keep only last 5 actions
        return " -> ".join(actions[-5:])

    def _extract_observation_key(self, observation: str) -> str:
        """
        Extract key information from observation for state identification.

        For WebShop: page type (search, product, etc.)
        For ALFWorld: location and items
        """
        obs_lower = observation.lower()

        # WebShop specific
        if "back to search" in obs_lower:
            if "buy now" in obs_lower:
                return "product_detail"
            else:
                return "search_results"
        elif "websshop" in obs_lower or "search" in obs_lower:
            return "search_page"

        # ALFWorld specific
        if "you see" in obs_lower or "you arrive" in obs_lower:
            # Extract location hint
            return observation[:100]

        # Default: use truncated observation
        return observation[:100]

    def add_or_get_state(
        self,
        task_id: int,
        step: int,
        observation: str,
        history: List[Dict],
        is_terminal: bool = False,
        terminal_reward: Optional[float] = None,
    ) -> StateNode:
        """
        Add a new state or get existing state node.

        Args:
            task_id: Task identifier
            step: Step number in trajectory
            observation: Environment observation
            history: Conversation history up to this point
            is_terminal: Whether this is a terminal state
            terminal_reward: Reward if terminal

        Returns:
            StateNode object (new or existing)
        """
        history_summary = self._extract_history_summary(history)
        state_id = self._compute_state_id(task_id, observation, history_summary)

        if state_id in self.nodes:
            # Update visit information
            node = self.nodes[state_id]
            node.visit_count += 1
            node.last_visit_time = self.current_time
            self.current_time += 1

            # Update terminal info if needed
            if is_terminal and not node.is_terminal:
                node.is_terminal = True
                node.terminal_reward = terminal_reward
                self.terminal_states.add(state_id)

            return node

        # Create new node
        node = StateNode(
            state_id=state_id,
            task_id=task_id,
            step=step,
            observation=observation,
            history_summary=history_summary,
            is_terminal=is_terminal,
            terminal_reward=terminal_reward,
            visit_count=1,
            last_visit_time=self.current_time,
        )

        self.nodes[state_id] = node
        self.task_states[task_id].add(state_id)

        if is_terminal:
            self.terminal_states.add(state_id)

        self.current_time += 1

        # Check if pruning is needed
        if len(self.nodes) > self.max_nodes:
            self._prune_graph()

        return node

    def add_transition(
        self,
        source_state: StateNode,
        action: str,
        target_state: StateNode,
        immediate_reward: float,
    ) -> ActionEdge:
        """
        Add a state transition edge.

        Args:
            source_state: Source state node
            action: Action taken (agent's output)
            target_state: Target state node
            immediate_reward: Immediate reward r(s, a)

        Returns:
            ActionEdge object (new or updated)
        """
        action_hash = hashlib.md5(action.encode()).hexdigest()[:16]

        # Check if transition already exists
        existing_edges = self.edges[source_state.state_id][action_hash]
        for edge in existing_edges:
            if edge.target_id == target_state.state_id:
                # Update existing edge
                edge.transition_count += 1
                # Incremental average of immediate reward
                n = edge.transition_count
                edge.immediate_reward = (
                    edge.immediate_reward * (n - 1) + immediate_reward
                ) / n
                return edge

        # Create new edge
        edge = ActionEdge(
            source_id=source_state.state_id,
            action=action,
            target_id=target_state.state_id,
            immediate_reward=immediate_reward,
            transition_count=1,
            q_value=immediate_reward,  # Initialize Q-value to immediate reward
        )

        self.edges[source_state.state_id][action_hash].append(edge)
        # 维护反向索引
        self.incoming_edges[target_state.state_id].add(source_state.state_id)
        return edge

    def add_trajectory(
        self,
        task_id: int,
        trajectory: List[Dict],
        final_reward: float,
        is_successful: bool,
    ) -> List[Tuple[StateNode, ActionEdge]]:
        """
        Add a complete trajectory to the graph.

        Args:
            task_id: Task identifier
            trajectory: List of {observation, action, reward} dicts
                        where reward is the reward received after executing action
            final_reward: Final trajectory reward
            is_successful: Whether the trajectory succeeded

        Returns:
            List of (StateNode, ActionEdge) tuples added
        """
        result = []
        history = []
        prev_state = None
        prev_action = None
        prev_reward = 0  # 用于正确对齐 reward
        state_sequence = []

        self.total_trajectories += 1
        if is_successful:
            self.successful_trajectories += 1

        for step, trans in enumerate(trajectory):
            observation = trans.get('observation', '')
            action = trans.get('action', '')
            step_reward = trans.get('reward', 0)
            is_last = (step == len(trajectory) - 1)

            # Add current state
            current_state = self.add_or_get_state(
                task_id=task_id,
                step=step,
                observation=observation,
                history=history.copy(),
                is_terminal=is_last,
                terminal_reward=final_reward if is_last else None,
            )
            state_sequence.append(current_state.state_id)

            # Update history
            history.append({'role': 'user', 'content': observation})
            history.append({'role': 'assistant', 'content': action})

            # Add transition edge: edge(s_{i-1}, a_{i-1} -> s_i) 使用 prev_reward
            # prev_reward 是执行 prev_action 后获得的奖励，即 trajectory[i-1].reward
            if prev_state is not None and prev_action is not None:
                edge = self.add_transition(
                    source_state=prev_state,
                    action=prev_action,
                    target_state=current_state,
                    immediate_reward=prev_reward,
                )
                result.append((prev_state, edge))

            prev_state = current_state
            prev_action = action
            prev_reward = step_reward  # 保存当前 reward 供下一步使用

        # Record successful path
        if is_successful and state_sequence:
            self.successful_paths.append(state_sequence)
            # Keep only last 1000 successful paths
            if len(self.successful_paths) > 1000:
                self.successful_paths = self.successful_paths[-1000:]

        return result

    def get_outgoing_edges(self, state_id: str) -> List[ActionEdge]:
        """Get all outgoing edges from a state"""
        all_edges = []
        for action_edges in self.edges.get(state_id, {}).values():
            all_edges.extend(action_edges)
        return all_edges

    def get_best_action(self, state_id: str) -> Optional[Tuple[str, float]]:
        """
        Get the best action and its Q-value from a state.

        Returns:
            (action, q_value) tuple or None if no actions available
        """
        edges = self.get_outgoing_edges(state_id)
        if not edges:
            return None
        best_edge = max(edges, key=lambda e: e.q_value)
        return (best_edge.action, best_edge.q_value)

    def get_state_value(self, state_id: str) -> float:
        """
        Get V(s) = max_a Q(s, a)

        For terminal states, returns terminal reward.
        """
        if state_id in self.terminal_states:
            node = self.nodes.get(state_id)
            if node and node.terminal_reward is not None:
                return node.terminal_reward
            return 0

        edges = self.get_outgoing_edges(state_id)
        if not edges:
            return 0
        return max(e.q_value for e in edges)

    def get_action_by_hash(self, state_id: str, action_hash: str) -> Optional[ActionEdge]:
        """Get edge by action hash"""
        edges = self.edges.get(state_id, {}).get(action_hash, [])
        return edges[0] if edges else None

    def _prune_graph(self):
        """
        LRU-based pruning: Remove low-visit, non-critical nodes.

        Keeps nodes that are part of successful paths.
        """
        # Mark critical nodes (on successful paths)
        critical_nodes = set()
        for path in self.successful_paths:
            critical_nodes.update(path)

        # Sort removable nodes by visit time
        removable = [
            (state_id, node)
            for state_id, node in self.nodes.items()
            if state_id not in critical_nodes
        ]
        removable.sort(key=lambda x: x[1].last_visit_time)

        # Remove oldest 20% of nodes
        num_to_remove = len(self.nodes) - int(self.max_nodes * 0.8)
        for i in range(min(num_to_remove, len(removable))):
            state_id = removable[i][0]
            self._remove_state(state_id)

    def _remove_state(self, state_id: str):
        """Remove a state node and its related edges"""
        if state_id not in self.nodes:
            return

        node = self.nodes[state_id]

        # Remove outgoing edges and update incoming_edges of targets
        if state_id in self.edges:
            for action_hash, edges in self.edges[state_id].items():
                for edge in edges:
                    # 从 target 的反向索引中移除 source
                    if edge.target_id in self.incoming_edges:
                        self.incoming_edges[edge.target_id].discard(state_id)
            del self.edges[state_id]

        # Remove incoming edges (使用反向索引，O(入度) 而不是 O(E))
        for source_id in self.incoming_edges.get(state_id, set()).copy():
            if source_id in self.edges:
                for action_hash in list(self.edges[source_id].keys()):
                    self.edges[source_id][action_hash] = [
                        e for e in self.edges[source_id][action_hash]
                        if e.target_id != state_id
                    ]
                    # Clean up empty lists
                    if not self.edges[source_id][action_hash]:
                        del self.edges[source_id][action_hash]

        # Remove from incoming_edges index
        if state_id in self.incoming_edges:
            del self.incoming_edges[state_id]

        # Remove node from indexes
        del self.nodes[state_id]
        self.task_states[node.task_id].discard(state_id)
        self.terminal_states.discard(state_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        total_edges = sum(
            len(edges)
            for action_dict in self.edges.values()
            for edges in action_dict.values()
        )

        return {
            'num_nodes': len(self.nodes),
            'num_edges': total_edges,
            'num_terminal_states': len(self.terminal_states),
            'num_tasks': len(self.task_states),
            'total_trajectories': self.total_trajectories,
            'successful_trajectories': self.successful_trajectories,
            'num_successful_paths': len(self.successful_paths),
        }

    def save(self, filepath: str):
        """Save graph to file"""
        # Count edges
        total_edges = sum(
            len(edges)
            for action_dict in self.edges.values()
            for edges in action_dict.values()
        )
        print(f"Saving graph: {len(self.nodes)} nodes, {total_edges} edges, "
              f"{len(self.terminal_states)} terminal states, "
              f"{self.successful_trajectories}/{self.total_trajectories} successful trajectories")

        data = {
            'task_name': self.task_name,
            'max_nodes': self.max_nodes,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'current_time': self.current_time,
            'total_trajectories': self.total_trajectories,
            'successful_trajectories': self.successful_trajectories,
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': self._edges_to_dict(),
            'terminal_states': list(self.terminal_states),
            'successful_paths': self.successful_paths[-100:],  # Keep last 100
            'task_states': {str(k): list(v) for k, v in self.task_states.items()},
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'StateActionGraph':
        """Load graph from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        graph = cls(
            task_name=data.get('task_name', 'webshop'),
            max_nodes=data.get('max_nodes', 10000),
            gamma=data.get('gamma', 0.95),
            alpha=data.get('alpha', 0.1),
        )

        graph.current_time = data.get('current_time', 0)
        graph.total_trajectories = data.get('total_trajectories', 0)
        graph.successful_trajectories = data.get('successful_trajectories', 0)

        # Restore nodes
        for state_id, node_data in data.get('nodes', {}).items():
            graph.nodes[state_id] = StateNode.from_dict(node_data)

        # Restore edges
        graph._edges_from_dict(data.get('edges', {}))

        # Restore indexes
        graph.terminal_states = set(data.get('terminal_states', []))
        graph.successful_paths = data.get('successful_paths', [])

        for task_id_str, state_ids in data.get('task_states', {}).items():
            graph.task_states[int(task_id_str)] = set(state_ids)

        # Count edges for logging
        total_edges = sum(
            len(edges)
            for action_dict in graph.edges.values()
            for edges in action_dict.values()
        )
        print(f"Loaded graph: {len(graph.nodes)} nodes, {total_edges} edges, "
              f"{len(graph.terminal_states)} terminal states, "
              f"{len(graph.incoming_edges)} nodes with incoming edges")

        return graph

    def _edges_to_dict(self) -> Dict:
        """Serialize edges to dictionary"""
        result = {}
        for source_id, action_dict in self.edges.items():
            result[source_id] = {}
            for action_hash, edges in action_dict.items():
                result[source_id][action_hash] = [e.to_dict() for e in edges]
        return result

    def _edges_from_dict(self, data: Dict):
        """Deserialize edges from dictionary"""
        for source_id, action_dict in data.items():
            for action_hash, edges_data in action_dict.items():
                edges = [ActionEdge.from_dict(e) for e in edges_data]
                self.edges[source_id][action_hash] = edges
                # 重建反向索引
                for edge in edges:
                    self.incoming_edges[edge.target_id].add(source_id)

    def merge_from(self, other: 'StateActionGraph'):
        """
        Merge another graph into this one.

        Used for combining graphs from parallel workers.
        """
        # Merge nodes
        for state_id, node in other.nodes.items():
            if state_id not in self.nodes:
                self.nodes[state_id] = node
            else:
                # Merge visit counts
                self.nodes[state_id].visit_count += node.visit_count
                # Update terminal info
                if node.is_terminal:
                    self.nodes[state_id].is_terminal = True
                    if node.terminal_reward is not None:
                        self.nodes[state_id].terminal_reward = node.terminal_reward

        # Merge edges
        for source_id, action_dict in other.edges.items():
            for action_hash, edges in action_dict.items():
                for edge in edges:
                    existing_edges = self.edges[source_id][action_hash]
                    found = False
                    for e in existing_edges:
                        if e.target_id == edge.target_id:
                            # Merge statistics
                            total_count = e.transition_count + edge.transition_count
                            e.immediate_reward = (
                                e.immediate_reward * e.transition_count +
                                edge.immediate_reward * edge.transition_count
                            ) / total_count
                            e.q_value = (e.q_value + edge.q_value) / 2
                            e.transition_count = total_count
                            found = True
                            break

                    if not found:
                        existing_edges.append(edge)
                        # 维护反向索引
                        self.incoming_edges[edge.target_id].add(source_id)

        # Merge indexes
        self.terminal_states.update(other.terminal_states)
        self.successful_paths.extend(other.successful_paths)

        for task_id, state_ids in other.task_states.items():
            self.task_states[task_id].update(state_ids)

        # Update statistics
        self.total_trajectories += other.total_trajectories
        self.successful_trajectories += other.successful_trajectories

        # Prune if needed
        if len(self.nodes) > self.max_nodes:
            self._prune_graph()
