"""
Value Propagation: Graph-based reward propagation algorithms

This module implements value propagation algorithms that replace
the simple Monte Carlo averaging in original IPR.
"""

import numpy as np
import hashlib
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state_graph import StateActionGraph, ActionEdge


class ValuePropagator:
    """
    Graph-based value propagation using TD-learning and Value Iteration.

    Replaces the Monte Carlo averaging in original IPR with more
    sophisticated value estimation methods.
    """

    def __init__(
        self,
        gamma: float = 0.95,
        alpha: float = 0.1,
        use_adaptive_alpha: bool = True,
    ):
        """
        Initialize value propagator.

        Args:
            gamma: Discount factor (0.95 for WebShop, 0.99 for ALFWorld)
            alpha: Base learning rate
            use_adaptive_alpha: Whether to use 1/n(s,a) as learning rate
        """
        self.gamma = gamma
        self.base_alpha = alpha
        self.use_adaptive_alpha = use_adaptive_alpha

    def incremental_update(
        self,
        graph: 'StateActionGraph',
        source_id: str,
        action: str,
        target_id: str,
        immediate_reward: float,
    ):
        """
        Incremental Q-Learning style update.

        Called after each transition for low-latency updates.

        Formula: Q(s,a) <- Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        Args:
            graph: The state-action graph
            source_id: Source state ID
            action: Action taken
            target_id: Target state ID
            immediate_reward: Immediate reward received
        """
        # Get target state V-value
        target_v = graph.get_state_value(target_id)

        # Find the edge
        edge = self._find_edge(graph, source_id, action, target_id)
        if edge is None:
            return

        # Compute learning rate
        if self.use_adaptive_alpha:
            # Adaptive learning rate: 1/n(s,a)
            alpha = 1.0 / max(edge.transition_count, 1)
        else:
            alpha = self.base_alpha

        # TD target
        td_target = immediate_reward + self.gamma * target_v

        # Q-value update
        td_error = td_target - edge.q_value
        edge.q_value = edge.q_value + alpha * td_error

    def batch_value_iteration(
        self,
        graph: 'StateActionGraph',
        num_iterations: int = 10,
        convergence_threshold: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Batch Value Iteration.

        Run at the end of each sampling round to propagate reward
        signals from terminal states to all reachable states.

        Args:
            graph: The state-action graph
            num_iterations: Maximum number of iterations
            convergence_threshold: Stop if max delta < threshold

        Returns:
            Dictionary of state V-values
        """
        if not graph.nodes:
            return {}

        # Initialize V-values
        v_values = {
            state_id: graph.get_state_value(state_id)
            for state_id in graph.nodes
        }

        for iteration in range(num_iterations):
            max_delta = 0

            # Shuffle state order for better convergence
            state_ids = list(graph.nodes.keys())
            np.random.shuffle(state_ids)

            for state_id in state_ids:
                node = graph.nodes[state_id]

                # Terminal states don't update
                if node.is_terminal:
                    continue

                # Get all outgoing edges
                edges = graph.get_outgoing_edges(state_id)
                if not edges:
                    continue

                # Update Q-value for each edge
                for edge in edges:
                    # Get successor state V-value
                    next_v = v_values.get(edge.target_id, 0)

                    # Q-value update: Q(s,a) = r(s,a) + γ * V(s')
                    new_q = edge.immediate_reward + self.gamma * next_v

                    delta = abs(new_q - edge.q_value)
                    max_delta = max(max_delta, delta)

                    edge.q_value = new_q

                # Update V-value: V(s) = max_a Q(s,a)
                old_v = v_values[state_id]
                new_v = max(e.q_value for e in edges)
                v_values[state_id] = new_v
                max_delta = max(max_delta, abs(new_v - old_v))

            # Convergence check
            if max_delta < convergence_threshold:
                print(f"Value iteration converged after {iteration + 1} iterations (max_delta={max_delta:.6f})")
                break

        return v_values

    def backward_propagation(
        self,
        graph: 'StateActionGraph',
        trajectory_state_ids: List[str],
        final_reward: float,
    ):
        """
        Backward propagation from terminal state.

        Quickly updates Q-values along a single trajectory.

        Args:
            graph: The state-action graph
            trajectory_state_ids: Sequence of state IDs in the trajectory
            final_reward: Final reward of the trajectory
        """
        if len(trajectory_state_ids) < 2:
            return

        # Propagate backwards
        cumulative_reward = final_reward

        for i in range(len(trajectory_state_ids) - 1, 0, -1):
            current_id = trajectory_state_ids[i]
            prev_id = trajectory_state_ids[i - 1]

            # Find the connecting edge
            edges = graph.get_outgoing_edges(prev_id)
            for edge in edges:
                if edge.target_id == current_id:
                    # Update Q-value
                    old_q = edge.q_value
                    new_q = edge.immediate_reward + self.gamma * cumulative_reward

                    # Incremental update with adaptive alpha
                    alpha = 1.0 / (edge.transition_count + 1)
                    edge.q_value = old_q + alpha * (new_q - old_q)

                    # Update cumulative reward for next step
                    cumulative_reward = edge.q_value
                    break

    def prioritized_sweeping(
        self,
        graph: 'StateActionGraph',
        changed_states: List[str],
        max_updates: int = 100,
        threshold: float = 0.01,
    ):
        """
        Prioritized sweeping for efficient value propagation.

        Only updates states that are likely to have changed significantly.

        Args:
            graph: The state-action graph
            changed_states: List of states that were recently updated
            max_updates: Maximum number of state updates
            threshold: Minimum priority to consider
        """
        import heapq

        # Priority queue: (-priority, state_id)
        # Negative because heapq is min-heap
        priority_queue = []

        # Initialize with changed states
        for state_id in changed_states:
            v_new = graph.get_state_value(state_id)
            heapq.heappush(priority_queue, (-abs(v_new), state_id))

        updated = set()
        num_updates = 0

        while priority_queue and num_updates < max_updates:
            neg_priority, state_id = heapq.heappop(priority_queue)
            priority = -neg_priority

            if priority < threshold:
                break

            if state_id in updated:
                continue

            updated.add(state_id)
            num_updates += 1

            # Update this state's Q-values
            edges = graph.get_outgoing_edges(state_id)
            for edge in edges:
                next_v = graph.get_state_value(edge.target_id)
                new_q = edge.immediate_reward + self.gamma * next_v
                edge.q_value = new_q

            # Find predecessors and add to queue
            for source_id, action_dict in graph.edges.items():
                for action_hash, edges in action_dict.items():
                    for edge in edges:
                        if edge.target_id == state_id:
                            # Compute priority for predecessor
                            pred_v_old = graph.get_state_value(source_id)
                            # Temporarily update
                            pred_edges = graph.get_outgoing_edges(source_id)
                            if pred_edges:
                                pred_v_new = max(e.q_value for e in pred_edges)
                                pred_priority = abs(pred_v_new - pred_v_old)
                                if pred_priority >= threshold:
                                    heapq.heappush(priority_queue, (-pred_priority, source_id))

    def compute_trajectory_q(
        self,
        graph: 'StateActionGraph',
        conversations: List[Dict],
        task_id: int,
    ) -> float:
        """
        Compute cumulative Q-value for a trajectory.

        Used for comparing trajectories in preference construction.

        Args:
            graph: The state-action graph
            conversations: Conversation history
            task_id: Task identifier

        Returns:
            Cumulative Q-value along the trajectory
        """
        total_q = 0
        history = []

        for i, msg in enumerate(conversations):
            role = msg.get('role') or ('user' if msg.get('from') == 'human' else 'assistant')
            content = msg.get('content') or msg.get('value', '')

            if role == 'user':
                observation = content
                # Compute state ID
                history_summary = graph._extract_history_summary(history)
                state_id = graph._compute_state_id(task_id, observation, history_summary)

                # Get state V-value
                v = graph.get_state_value(state_id)
                total_q += v

            history.append({'role': role, 'content': content})

        return total_q

    def get_step_q(
        self,
        graph: 'StateActionGraph',
        step_conversations: List[Dict],
        task_id: int,
    ) -> float:
        """
        Get Q-value for a specific step.

        Args:
            graph: The state-action graph
            step_conversations: Conversations up to this step
            task_id: Task identifier

        Returns:
            Q-value for the last state in the conversation
        """
        if not step_conversations:
            return 0

        history = []
        last_observation = None

        for msg in step_conversations:
            role = msg.get('role') or ('user' if msg.get('from') == 'human' else 'assistant')
            content = msg.get('content') or msg.get('value', '')

            if role == 'user':
                last_observation = content

            history.append({'role': role, 'content': content})

        if last_observation is None:
            return 0

        # Compute state ID
        history_summary = graph._extract_history_summary(history[:-2] if len(history) >= 2 else [])
        state_id = graph._compute_state_id(task_id, last_observation, history_summary)

        return graph.get_state_value(state_id)

    def _find_edge(
        self,
        graph: 'StateActionGraph',
        source_id: str,
        action: str,
        target_id: str,
    ) -> Optional['ActionEdge']:
        """Find a specific edge in the graph"""
        action_hash = hashlib.md5(action.encode()).hexdigest()[:16]
        edges = graph.edges.get(source_id, {}).get(action_hash, [])
        for edge in edges:
            if edge.target_id == target_id:
                return edge
        return None
