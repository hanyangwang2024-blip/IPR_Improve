"""
Trajectory Stitching: Find optimal paths and build preference data

This module implements trajectory stitching that enables super-expert
performance by finding paths in the graph that are better than the
expert trajectory.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    from .state_graph import StateActionGraph, StateNode


class TrajectoryStitcher:
    """
    Trajectory stitching: Find optimal paths in the state graph.

    This enables the agent to learn from synthesized trajectories
    that may be better than any single observed trajectory.
    """

    def __init__(
        self,
        graph: 'StateActionGraph',
        max_path_length: int = 15,
    ):
        """
        Initialize trajectory stitcher.

        Args:
            graph: The state-action graph
            max_path_length: Maximum path length to search
        """
        self.graph = graph
        self.max_path_length = max_path_length

    def find_best_path(
        self,
        start_state_id: str,
        task_id: int,
    ) -> Optional[List[Tuple[str, str, float]]]:
        """
        Find the best path from a given state using greedy search.

        Args:
            start_state_id: Starting state ID
            task_id: Task ID (ensures path stays within same task)

        Returns:
            List of (state_id, action, q_value) tuples, or None if no path found
        """
        if start_state_id not in self.graph.nodes:
            return None

        path = []
        current_id = start_state_id
        visited = {current_id}

        for _ in range(self.max_path_length):
            node = self.graph.nodes.get(current_id)
            if node is None or node.is_terminal:
                break

            # Ensure we stay within the same task
            if node.task_id != task_id:
                break

            # Get best action
            best_action_info = self.graph.get_best_action(current_id)
            if best_action_info is None:
                break

            action, q_value = best_action_info

            # Find target state
            edges = self.graph.get_outgoing_edges(current_id)
            best_edge = None
            for e in edges:
                if e.action == action:
                    if best_edge is None or e.q_value > best_edge.q_value:
                        best_edge = e

            if best_edge is None:
                break

            # Check for cycles
            if best_edge.target_id in visited:
                break

            path.append((current_id, action, q_value))
            visited.add(best_edge.target_id)
            current_id = best_edge.target_id

        return path if path else None

    def find_best_path_beam_search(
        self,
        start_state_id: str,
        task_id: int,
        beam_width: int = 3,
    ) -> Optional[List[Tuple[str, str, float]]]:
        """
        Find best path using beam search for better exploration.

        Args:
            start_state_id: Starting state ID
            task_id: Task ID
            beam_width: Number of candidates to keep at each step

        Returns:
            Best path found, or None
        """
        if start_state_id not in self.graph.nodes:
            return None

        # Beam: List of (cumulative_q, path, current_state, visited)
        beam = [(0.0, [], start_state_id, {start_state_id})]

        for _ in range(self.max_path_length):
            candidates = []

            for cum_q, path, current_id, visited in beam:
                node = self.graph.nodes.get(current_id)
                if node is None or node.is_terminal:
                    # Keep terminal paths as candidates
                    candidates.append((cum_q, path, current_id, visited, True))
                    continue

                if node.task_id != task_id:
                    continue

                edges = self.graph.get_outgoing_edges(current_id)
                for edge in edges:
                    if edge.target_id in visited:
                        continue

                    new_path = path + [(current_id, edge.action, edge.q_value)]
                    new_visited = visited | {edge.target_id}
                    new_cum_q = cum_q + edge.q_value

                    candidates.append((new_cum_q, new_path, edge.target_id, new_visited, False))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: -x[0])
            beam = [(c[0], c[1], c[2], c[3]) for c in candidates[:beam_width] if not c[4]]

            # If all candidates are terminal, we're done
            if not beam:
                # Return best terminal path
                terminal_candidates = [c for c in candidates if c[4]]
                if terminal_candidates:
                    best = max(terminal_candidates, key=lambda x: x[0])
                    return best[1] if best[1] else None
                break

        # Return best path from beam
        if beam:
            best = max(beam, key=lambda x: x[0])
            return best[1] if best[1] else None
        return None

    def check_reachability(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
    ) -> bool:
        """
        Check if target is reachable from source using BFS.

        Ensures trajectory stitching only uses actually reachable paths.

        Args:
            source_id: Source state ID
            target_id: Target state ID
            max_depth: Maximum search depth

        Returns:
            True if reachable, False otherwise
        """
        if source_id == target_id:
            return True

        if source_id not in self.graph.nodes:
            return False

        # BFS
        queue = deque([(source_id, 0)])
        visited = {source_id}

        while queue:
            current_id, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for edge in self.graph.get_outgoing_edges(current_id):
                if edge.target_id == target_id:
                    return True

                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, depth + 1))

        return False

    def compute_path_q(self, path: List[Tuple[str, str, float]]) -> float:
        """Compute cumulative Q-value for a path"""
        return sum(q for _, _, q in path)

    def path_to_conversations(
        self,
        path: List[Tuple[str, str, float]],
    ) -> List[Dict]:
        """
        Convert a graph path to conversation format.

        Args:
            path: List of (state_id, action, q_value)

        Returns:
            List of conversation messages
        """
        conversations = []
        for state_id, action, _ in path:
            node = self.graph.nodes.get(state_id)
            if node:
                conversations.append({
                    "from": "human",
                    "value": node.observation
                })
                conversations.append({
                    "from": "gpt",
                    "value": action
                })
        return conversations


class GraphPreferenceBuilder:
    """
    Build preference data using graph-based Q-values and trajectory stitching.

    Replaces the original MC-based preference construction with:
    1. Q-values from graph instead of MC averaging
    2. Trajectory stitching for super-expert paths
    """

    def __init__(
        self,
        graph: 'StateActionGraph',
        step_threshold: float = 0.1,
        traj_threshold: float = 0.05,
        stitching_threshold: float = 1.1,  # Path must be 10% better
    ):
        """
        Initialize preference builder.

        Args:
            graph: The state-action graph
            step_threshold: Threshold for step-level preferences
            traj_threshold: Threshold for trajectory-level preferences
            stitching_threshold: Multiplier threshold for stitching (1.1 = 10% better)
        """
        self.graph = graph
        self.stitcher = TrajectoryStitcher(graph)
        self.step_threshold = step_threshold
        self.traj_threshold = traj_threshold
        self.stitching_threshold = stitching_threshold

    def build_preference_data(
        self,
        explored_trajectories: List[Dict],
        expert_data: Dict,
        instruction_part_len: int = 2,
        icl_prompt_part_len: int = 10,
        enable_stitching: bool = True,
    ) -> List[Dict]:
        """
        Build preference data using graph-based methods.

        Args:
            explored_trajectories: List of explored trajectory items
            expert_data: Dictionary of expert data keyed by "{id}_{iteration}"
            instruction_part_len: Length of instruction part in expert conversations
            icl_prompt_part_len: Length of ICL prompt part in agent conversations
            enable_stitching: Whether to enable trajectory stitching

        Returns:
            List of preference data items
        """
        pm_data = []
        stats = {
            'global_traj': 0,
            'local_step': 0,
            'local_traj': 0,
            'stitching': 0,
            'win': 0,
            'tie': 0,
            'lose': 0,
        }

        for item in explored_trajectories:
            task_id = item.get('id')
            iteration = item.get('iteration', 0)
            expert_key = f"{task_id}_{iteration}"

            if expert_key not in expert_data:
                continue

            expert_info = expert_data[expert_key]

            # Get Q-values from graph
            agent_q = self._compute_trajectory_q(
                item.get('agent_conversations', []),
                task_id
            )
            expert_q = self._compute_trajectory_q(
                expert_info.get('gpt_conversations', []),
                task_id
            )

            # Also consider final rewards
            agent_final = item.get('agent_final_reward', 0)
            expert_final = expert_info.get('gpt_reward', 0)

            # Combined score
            agent_score = 0.5 * agent_q + 0.5 * agent_final
            expert_score = 0.5 * expert_q + 0.5 * expert_final

            # Global trajectory preference (iteration=0)
            if iteration == 0:
                pref = self._build_global_preference(
                    item, expert_info, agent_score, expert_score,
                    instruction_part_len, icl_prompt_part_len
                )
                if pref:
                    pm_data.append(pref)
                    stats['global_traj'] += 1
                    if agent_score > expert_score:
                        stats['win'] += 1
                    else:
                        stats['lose'] += 1
                else:
                    stats['tie'] += 1

            # Local trajectory preference (iteration>0)
            else:
                pref = self._build_local_preference(
                    item, expert_info, task_id,
                    instruction_part_len, icl_prompt_part_len
                )
                if pref:
                    pm_data.append(pref)
                    if pref.get('type') == 'step':
                        stats['local_step'] += 1
                    else:
                        stats['local_traj'] += 1

            # Trajectory stitching preference
            if enable_stitching:
                stitch_pref = self._build_stitching_preference(
                    item, expert_info, task_id,
                    instruction_part_len
                )
                if stitch_pref:
                    pm_data.append(stitch_pref)
                    stats['stitching'] += 1

        print(f"Built {len(pm_data)} preference pairs:")
        print(f"  - Global trajectory: {stats['global_traj']}")
        print(f"  - Local step: {stats['local_step']}")
        print(f"  - Local trajectory: {stats['local_traj']}")
        print(f"  - Stitching: {stats['stitching']}")
        print(f"  - Win/Tie/Lose: {stats['win']}/{stats['tie']}/{stats['lose']}")

        return pm_data

    def _compute_trajectory_q(
        self,
        conversations: List[Dict],
        task_id: int,
    ) -> float:
        """Compute cumulative Q-value for a trajectory"""
        total_q = 0
        history = []

        for msg in conversations:
            role = msg.get('role') or ('user' if msg.get('from') == 'human' else 'assistant')
            content = msg.get('content') or msg.get('value', '')

            if role == 'user':
                observation = content
                history_summary = self.graph._extract_history_summary(history)
                state_id = self.graph._compute_state_id(task_id, observation, history_summary)
                v = self.graph.get_state_value(state_id)
                total_q += v

            history.append({'role': role, 'content': content})

        return total_q

    def _build_global_preference(
        self,
        agent_item: Dict,
        expert_info: Dict,
        agent_score: float,
        expert_score: float,
        instruction_part_len: int,
        icl_prompt_part_len: int,
    ) -> Optional[Dict]:
        """Build global trajectory preference"""
        if abs(agent_score - expert_score) < self.traj_threshold:
            return None

        agent_convs = self._template_change(agent_item.get('agent_conversations', []))
        expert_convs = expert_info.get('gpt_conversations', [])

        if self._is_empty_conversations(agent_convs) or self._is_empty_conversations(expert_convs):
            return None

        task_id = agent_item.get('id')
        iteration = agent_item.get('iteration', 0)

        if agent_score > expert_score:
            return {
                "id": int(f"{task_id}{iteration}"),
                "prompt": expert_convs[:instruction_part_len + 1],
                "chosen": agent_convs[icl_prompt_part_len + 1: -1] if len(agent_convs) > icl_prompt_part_len + 1 else agent_convs,
                "rejected": expert_convs[instruction_part_len + 1:],
                "agent_score": agent_score,
                "expert_score": expert_score,
                "type": "global",
            }
        else:
            return {
                "id": int(f"{task_id}{iteration}"),
                "prompt": expert_convs[:instruction_part_len + 1],
                "chosen": expert_convs[instruction_part_len + 1:],
                "rejected": agent_convs[icl_prompt_part_len + 1: -1] if len(agent_convs) > icl_prompt_part_len + 1 else agent_convs,
                "agent_score": agent_score,
                "expert_score": expert_score,
                "type": "global",
            }

    def _build_local_preference(
        self,
        agent_item: Dict,
        expert_info: Dict,
        task_id: int,
        instruction_part_len: int,
        icl_prompt_part_len: int,
    ) -> Optional[Dict]:
        """Build local step/trajectory preference"""
        agent_step_convs = self._template_change(agent_item.get('agent_step_conversations', []))
        expert_step_convs = expert_info.get('gpt_step_conversations', [])

        if not agent_step_convs or not expert_step_convs:
            return None

        if self._is_empty_conversations(agent_step_convs) or self._is_empty_conversations(expert_step_convs):
            return None

        # Get step-level Q-values
        agent_step_q = self._get_step_q(agent_step_convs, task_id)
        expert_step_q = self._get_step_q(expert_step_convs, task_id)

        # Also consider MC rewards if available
        agent_mc = agent_item.get('monte_carlo_step_reward', agent_item.get('agent_step_reward', 0))
        expert_mc = expert_info.get('monte_carlo_step_reward', expert_info.get('gpt_step_reward', 0))

        # Combined step score
        agent_step_score = 0.5 * agent_step_q + 0.5 * agent_mc
        expert_step_score = 0.5 * expert_step_q + 0.5 * expert_mc

        if abs(agent_step_score - expert_step_score) < self.step_threshold:
            return None

        iteration = agent_item.get('iteration', 0)
        gpt_length = len(expert_step_convs)
        agent_length = gpt_length - instruction_part_len + icl_prompt_part_len

        if agent_step_score > expert_step_score:
            return {
                "id": int(f"{task_id}{iteration}"),
                "prompt": expert_step_convs[:gpt_length - 1],
                "chosen": [agent_step_convs[agent_length - 1]] if agent_length > 0 and agent_length <= len(agent_step_convs) else [],
                "rejected": [expert_step_convs[gpt_length - 1]],
                "step_q_diff": agent_step_score - expert_step_score,
                "type": "step",
            }
        else:
            return {
                "id": int(f"{task_id}{iteration}"),
                "prompt": expert_step_convs[:gpt_length - 1],
                "chosen": [expert_step_convs[gpt_length - 1]],
                "rejected": [agent_step_convs[agent_length - 1]] if agent_length > 0 and agent_length <= len(agent_step_convs) else [],
                "step_q_diff": expert_step_score - agent_step_score,
                "type": "step",
            }

    def _build_stitching_preference(
        self,
        agent_item: Dict,
        expert_info: Dict,
        task_id: int,
        instruction_part_len: int,
    ) -> Optional[Dict]:
        """
        Build trajectory stitching preference.

        If the graph contains a path better than the expert, use it as chosen.
        """
        expert_convs = expert_info.get('gpt_conversations', [])
        if not expert_convs:
            return None

        # Convert expert trajectory to trajectory format (skip instruction part)
        # expert_convs[:instruction_part_len] 是 instruction 部分，跳过
        task_convs = expert_convs[instruction_part_len:]
        expert_traj = self._conversations_to_trajectory(task_convs)

        if not expert_traj:
            return None

        # Try to find a better path from each intermediate state
        history = []
        for i, trans in enumerate(expert_traj[:-1]):
            observation = trans['observation']

            # Build history for state computation
            history_summary = self.graph._extract_history_summary(history)
            state_id = self.graph._compute_state_id(task_id, observation, history_summary)

            # Find best path from this state
            best_path = self.stitcher.find_best_path(state_id, task_id)
            if not best_path:
                history.append({'role': 'user', 'content': observation})
                history.append({'role': 'assistant', 'content': trans['action']})
                continue

            # Compute path Q-value
            path_q = self.stitcher.compute_path_q(best_path)

            # Compute expert remaining Q-value (从 task_convs 的第 i 步开始)
            remaining_expert_q = self._compute_trajectory_q(task_convs[i * 2:], task_id)

            # If graph path is significantly better
            # 处理 expert Q 为 0 或接近 0 的情况，避免乘法阈值失效
            is_better = False
            if abs(remaining_expert_q) < 0.01:
                # 当 expert Q 接近 0 时，使用绝对阈值
                is_better = path_q > 0.1
            else:
                is_better = path_q > remaining_expert_q * self.stitching_threshold

            if is_better:
                # Build stitching preference
                # prompt = instruction + task_convs 前 i 步 (包含当前 observation)
                # i * 2 是因为每步有 (human, gpt) 两条消息
                # +1 是包含当前 observation (human)
                prefix_convs = expert_convs[:instruction_part_len + i * 2 + 1]
                graph_suffix = self.stitcher.path_to_conversations(best_path)

                iteration = agent_item.get('iteration', 0)

                return {
                    "id": int(f"{task_id}{iteration}_stitch_{i}"),
                    "prompt": prefix_convs,
                    "chosen": graph_suffix,
                    "rejected": task_convs[i * 2 + 1:],  # 从当前 gpt response 开始
                    "stitching_point": i,
                    "path_q": path_q,
                    "expert_q": remaining_expert_q,
                    "type": "stitching",
                }

            history.append({'role': 'user', 'content': observation})
            history.append({'role': 'assistant', 'content': trans['action']})

        return None

    def _conversations_to_trajectory(self, conversations: List[Dict]) -> List[Dict]:
        """Convert conversation format to trajectory format"""
        trajectory = []
        i = 0
        while i < len(conversations) - 1:
            user_msg = conversations[i]
            assistant_msg = conversations[i + 1] if i + 1 < len(conversations) else None

            user_content = user_msg.get('content') or user_msg.get('value', '')
            assistant_content = (assistant_msg.get('content') or assistant_msg.get('value', '')) if assistant_msg else ""

            # Check roles
            user_role = user_msg.get('role') or ('user' if user_msg.get('from') == 'human' else 'assistant')
            if user_role != 'user':
                i += 1
                continue

            trajectory.append({
                'observation': user_content,
                'action': assistant_content,
                'reward': 0,
            })
            i += 2

        return trajectory

    def _get_step_q(self, step_conversations: List, task_id: int) -> float:
        """Get Q-value for a specific step"""
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

        history_for_summary = history[:-2] if len(history) >= 2 else []
        history_summary = self.graph._extract_history_summary(history_for_summary)
        state_id = self.graph._compute_state_id(task_id, last_observation, history_summary)

        return self.graph.get_state_value(state_id)

    def _template_change(self, conversations: List[Dict]) -> List[Dict]:
        """Convert history format to conversation format"""
        messages = []
        for item in conversations:
            if isinstance(item, dict):
                if 'role' in item:
                    # Already in role format, convert to from/value format
                    messages.append({
                        'from': 'gpt' if item['role'] == 'assistant' else 'human',
                        'value': item.get('content', '').strip()
                    })
                elif 'from' in item:
                    messages.append(item)
        return messages

    def _is_empty_conversations(self, conversations: List[Dict]) -> bool:
        """Check if any conversation message is empty"""
        for item in conversations:
            value = item.get('value') or item.get('content', '')
            if value.strip() == "":
                return True
        return False
