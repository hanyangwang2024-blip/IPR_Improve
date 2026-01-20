"""
Graph-based Preference Construction for ALFWorld

This is the Graph-IPR version of construct_preference_monte_carlo_alfworld.py.
It uses graph-based Q-values instead of MC averaging and supports trajectory stitching.
"""

import os
import json
import argparse
import glob
import numpy as np
from typing import Dict, List, Any, Optional

from graph_ipr.state_graph import StateActionGraph
from graph_ipr.value_propagation import ValuePropagator
from graph_ipr.trajectory_stitching import GraphPreferenceBuilder


# Constants for ALFWorld
instruction_part = {
    "alfworld": [
        {
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ]
}

icl_prompt_part = {
    "alfworld": [
        {
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger."
        },
        {
            "from": "gpt",
            "value": "Thought: I need to place a toilet paper roll on the toilet paper hanger. First, I should check if there's already a roll on the hanger or if it's empty.\nAction: go to toiletpaperhanger 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toiletpaperhanger 1, you see nothing."
        },
        {
            "from": "gpt",
            "value": "Thought: The toilet paper hanger is empty, so I need to find a new roll of toilet paper to put on it.\nAction: go to toilet 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1."
        },
        {
            "from": "gpt",
            "value": "Thought: I've found a toilet paper roll on the toilet. I should pick it up so I can place it on the hanger.\nAction: take toiletpaper 1 from toilet 1"
        },
        {
            "from": "human",
            "value": "Observation: You pick up the toiletpaper 1 from the toilet 1."
        },
        {
            "from": "gpt",
            "value": "Thought: Now that I have the toilet paper roll, I need to return to the toilet paper hanger to install it.\nAction: go to toiletpaperhanger 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toiletpaperhanger 1, you see nothing."
        },
        {
            "from": "gpt",
            "value": "Thought: I'm back at the toilet paper hanger with the roll in hand. It's time to put the toilet paper on the hanger.\nAction: put toiletpaper 1 in/on toiletpaperhanger 1"
        }
    ]
}


def template_change(conversation: List[Dict]) -> List[Dict]:
    """Convert role format to from/value format"""
    messages = []
    for item in conversation:
        message = {}
        if item.get('role') == "assistant":
            message['from'] = "gpt"
            message['value'] = item.get('content', '').strip()
        else:
            message['from'] = "human"
            message['value'] = item.get('content', '').strip()
        messages.append(message)
    return messages


def is_empty_conversations(conversation: List[Dict]) -> bool:
    """Check if any message is empty"""
    for item in conversation:
        value = item.get('value') or item.get('content', '')
        if value.strip() == "":
            return True
    return False


def cal_monte_carlo_reward(args) -> List[Dict]:
    """Original MC reward calculation"""
    traj_path = args.sample_path
    results = {}
    results_original_reward = {}

    # 递归搜索所有 JSON 文件，兼容新旧目录结构
    # 新结构: sample_path/sampled_traj_0/worker_*/alfworld_traj_*.json
    # 旧结构: sample_path/sampled_traj_*/alfworld_traj_*.json
    all_json_paths = glob.glob(f"{traj_path}/**/*.json", recursive=True)

    if not all_json_paths:
        print(f"Warning: No JSON files found in {traj_path}")

    for path in all_json_paths:
        try:
            cur_results = json.load(open(path))
            if not isinstance(cur_results, list):
                cur_results = [cur_results]
            for item in cur_results:
                if 'id' not in item or 'iteration' not in item:
                    continue
                id = item['id']
                iteration = item['iteration']
                id_iteration = f"{id}_{iteration}"
                if id_iteration not in results:
                    results[id_iteration] = [item.get('agent_final_reward', 0)]
                else:
                    results[id_iteration].append(item.get('agent_final_reward', 0))
                if id_iteration not in results_original_reward:
                    results_original_reward[id_iteration] = item.get('agent_step_reward', 0)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            continue

    final_results = {}
    for key, value in results.items():
        final_results[key] = {
            "monte_carlo_reward": np.mean(value),
            "env_reward": results_original_reward.get(key, 0)
        }

    output_data = []
    for file in glob.glob(f"{args.traj_path}/*.json"):
        data = json.load(open(file))
        for item in data:
            id = item['id']
            iteration = item['iteration']
            id_iteration = f"{id}_{iteration}"

            if id_iteration in final_results:
                item['monte_carlo_step_reward'] = final_results[id_iteration]['monte_carlo_reward']
            else:
                item['monte_carlo_step_reward'] = item.get('agent_step_reward', 0)
            output_data.append(item)

    return output_data


def build_preference_with_graph(args):
    """
    Build preference data using graph-based Q-values for ALFWorld.
    """
    # Load state graph
    graph_path = args.graph_path
    if not os.path.exists(graph_path):
        print(f"Warning: State graph not found at {graph_path}")
        print("Falling back to MC-based preference construction")
        return build_preference_mc(args)

    graph = StateActionGraph.load(graph_path)
    print(f"Loaded state graph with {len(graph.nodes)} nodes")

    # Load expert data
    golden_raw = json.load(open(f"data/sft_data_{args.task}.json"))

    # Load game_file to id mapping
    game_file_to_id = {}
    if os.path.exists("data/game_file_to_id.json"):
        game_file_to_id = json.load(open("data/game_file_to_id.json"))

    # Load explored trajectories
    explored_traj = cal_monte_carlo_reward(args)

    # Map game_file to task_id for ALFWorld
    for item in explored_traj:
        if 'game_file' in item:
            game_file = item['game_file']
            if game_file in game_file_to_id:
                item['id'] = game_file_to_id[game_file]

    # Create preference builder with ALFWorld-specific thresholds
    preference_builder = GraphPreferenceBuilder(
        graph=graph,
        step_threshold=args.step_threshold,
        traj_threshold=args.traj_threshold,
        stitching_threshold=1.1,
    )

    # Build preference data
    pm_data = preference_builder.build_preference_data(
        explored_trajectories=explored_traj,
        expert_data=golden_raw,
        instruction_part_len=len(instruction_part["alfworld"]),
        icl_prompt_part_len=len(icl_prompt_part["alfworld"]),
        enable_stitching=args.enable_stitching,
    )

    # Filter out invalid entries
    valid_pm_data = []
    for item in pm_data:
        if item.get('chosen') and item.get('rejected'):
            clean_item = {
                'id': item['id'],
                'prompt': item['prompt'],
                'chosen': item['chosen'],
                'rejected': item['rejected'],
            }
            valid_pm_data.append(clean_item)

    print(f"\nFinal valid preference pairs: {len(valid_pm_data)}")

    # Save
    json.dump(valid_pm_data, open(args.output_path, "w"), indent=4)
    print(f"Saved preference data to {args.output_path}")

    return valid_pm_data


def build_preference_mc(args):
    """
    Original MC-based preference construction for ALFWorld (fallback).
    """
    win = 0
    tie = 0
    lose = 0
    global_traj = 0
    local_step_traj = 0
    local_entire_traj = 0

    golden_raw = json.load(open(f"data/sft_data_{args.task}.json"))
    game_file_to_id = {}
    if os.path.exists("data/game_file_to_id.json"):
        game_file_to_id = json.load(open("data/game_file_to_id.json"))

    pm_data = []
    explored_traj = cal_monte_carlo_reward(args)

    step_monte_carlo_threshold = args.step_threshold
    traj_threshold = args.traj_threshold

    if args.global_traj:
        for item in explored_traj:
            game_file = item.get('game_file', '')
            id = game_file_to_id.get(game_file, item['id'])

            iteration = item['iteration']
            if iteration != 0:
                continue

            agent_final_reward = item['agent_final_reward']
            expert_key = f"{id}_0"
            if expert_key not in golden_raw:
                continue

            gpt_reward = golden_raw[expert_key]['gpt_reward']
            gpt_conversations = golden_raw[expert_key]['gpt_conversations']
            agent_conversations = template_change(item['agent_conversations'])

            if is_empty_conversations(agent_conversations) or is_empty_conversations(gpt_conversations):
                continue

            if agent_final_reward > gpt_reward + traj_threshold:
                win += 1
                global_traj += 1
                pm_data.append({
                    "id": int(f"{id}{iteration}"),
                    "prompt": gpt_conversations[:len(instruction_part["alfworld"])+1],
                    "chosen": agent_conversations[len(icl_prompt_part["alfworld"])+1: -1],
                    "rejected": gpt_conversations[len(instruction_part["alfworld"])+1:],
                })
            elif gpt_reward > agent_final_reward + traj_threshold:
                lose += 1
                global_traj += 1
                pm_data.append({
                    "id": int(f"{id}{iteration}"),
                    "prompt": gpt_conversations[:len(instruction_part["alfworld"])+1],
                    "chosen": gpt_conversations[len(instruction_part["alfworld"])+1:],
                    "rejected": agent_conversations[len(icl_prompt_part["alfworld"])+1: -1],
                })
            else:
                tie += 1

    if args.local_traj:
        for item in explored_traj:
            game_file = item.get('game_file', '')
            id = game_file_to_id.get(game_file, item['id'])

            iteration = item['iteration']
            if iteration == 0:
                continue

            expert_key = f"{id}_{iteration}"
            if expert_key not in golden_raw:
                continue

            agent_conversations = template_change(item['agent_conversations'])
            agent_step_reward = item['monte_carlo_step_reward']
            agent_final_reward = item['agent_final_reward']

            gpt_conversations = golden_raw[expert_key]['gpt_conversations']
            gpt_step_reward = golden_raw[expert_key].get('monte_carlo_step_reward', 0)
            gpt_final_reward = golden_raw[expert_key]['gpt_reward']
            gpt_step_conversations = golden_raw[expert_key]['gpt_step_conversations']

            gpt_length = len(gpt_step_conversations)
            agent_length = gpt_length - len(instruction_part["alfworld"]) + len(icl_prompt_part["alfworld"])

            if is_empty_conversations(agent_conversations) or is_empty_conversations(gpt_conversations):
                continue

            # ALFWorld special: prefer shorter trajectories
            flag = False
            if len(agent_conversations[agent_length-1: -1]) < len(gpt_conversations[gpt_length-1:]):
                flag = True

            if (agent_final_reward > gpt_final_reward + traj_threshold or flag) and agent_step_reward > gpt_step_reward + step_monte_carlo_threshold:
                win += 1
                local_entire_traj += 1
                pm_data.append({
                    "id": int(f"{id}{iteration}"),
                    "prompt": gpt_conversations[:gpt_length-1],
                    "chosen": agent_conversations[agent_length-1: -1],
                    "rejected": gpt_conversations[gpt_length-1:],
                })
            elif gpt_final_reward > agent_final_reward + traj_threshold and gpt_step_reward > agent_step_reward + step_monte_carlo_threshold:
                lose += 1
                local_entire_traj += 1
                pm_data.append({
                    "id": int(f"{id}{iteration}"),
                    "prompt": gpt_conversations[:gpt_length-1],
                    "chosen": gpt_conversations[gpt_length-1:],
                    "rejected": agent_conversations[agent_length-1: -1],
                })
            else:
                tie += 1

    json.dump(pm_data, open(args.output_path, "w"), indent=4)
    print(f"win: {win}, tie: {tie}, lose: {lose}")
    print(f"global_traj: {global_traj}, local_step_traj: {local_step_traj}, local_entire_traj: {local_entire_traj}")

    return pm_data


def main():
    parser = argparse.ArgumentParser("Graph-based Preference Construction for ALFWorld")

    parser.add_argument("--task", type=str, default="alfworld")
    parser.add_argument("--golden_traj_path", type=str, default="data/alfworld_sft.json")
    parser.add_argument("--output_path", type=str, default='test.json')
    parser.add_argument("--traj_path", type=str, default="")
    parser.add_argument("--step_action", action="store_true")
    parser.add_argument("--global_traj", action="store_true")
    parser.add_argument("--local_traj", action="store_true")
    parser.add_argument("--sample_path", type=str, default="")

    # ALFWorld-specific thresholds (lower than WebShop due to binary rewards)
    parser.add_argument("--step_threshold", type=float, default=0.05)
    parser.add_argument("--traj_threshold", type=float, default=0.01)

    # Graph-IPR specific
    parser.add_argument("--graph_path", type=str, default="")
    parser.add_argument("--enable_stitching", action="store_true")
    parser.add_argument("--use_graph", action="store_true")

    args = parser.parse_args()

    if args.use_graph and args.graph_path:
        build_preference_with_graph(args)
    else:
        if args.global_traj or args.local_traj:
            build_preference_mc(args)
        else:
            print("Please specify --global_traj or --local_traj, or use --use_graph with --graph_path")


if __name__ == "__main__":
    main()
