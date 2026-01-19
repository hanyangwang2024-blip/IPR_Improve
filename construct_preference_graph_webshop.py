"""
Graph-based Preference Construction for WebShop

This is the Graph-IPR version of construct_preference_monte_carlo_webshop.py.
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


# Constants (same as original)
instruction_part = {
    "webshop": [
        {
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
    ],
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
    "webshop": [
        {
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "Observation:\nWebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should use the search bar to look for the product I need.\nAction: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L'eau d'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L'eau D'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L'eau D'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L'Eau d'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L'Eau D'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L'eau D'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L'eau d'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L'eau d'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the product I need, which is B000VOHH8I.\nAction: click[B000VOHH8I]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.\nAction: click[6.76 fl oz (pack of 1)]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the 'Buy Now' button to purchase the product.\nAction: click[Buy Now]"
        }
    ],
    "alfworld": [
        {
            "from": "human",
            "value": "Interact with a household to solve a task..."
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        # ... (same as original)
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
    """
    Original MC reward calculation (kept for compatibility).

    Also loads graph-based Q-values if available.
    """
    traj_path = args.sample_path
    results = {}
    results_original_reward = {}

    sample_num = len(glob.glob(traj_path + "/*"))
    for i in range(sample_num):
        paths = glob.glob(f"{traj_path}/sampled_traj_{i}/*.json")
        cur_results = []
        for path in paths:
            cur_results.extend(json.load(open(path)))
        for item in cur_results:
            id = item['id']
            iteration = item['iteration']
            id_iteration = f"{id}_{iteration}"
            if id_iteration not in results:
                results[id_iteration] = [item['agent_final_reward']]
            else:
                results[id_iteration].append(item['agent_final_reward'])
            if id_iteration not in results_original_reward:
                results_original_reward[id_iteration] = item['agent_step_reward']
            else:
                assert results_original_reward[id_iteration] == item['agent_step_reward']

    final_results = {}
    for key, value in results.items():
        final_results[key] = {
            "monte_carlo_reward": np.mean(value),
            "env_reward": results_original_reward[key]
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
                item['monte_carlo_step_reward'] = item['agent_step_reward']
            output_data.append(item)

    return output_data


def build_preference_with_graph(args):
    """
    Build preference data using graph-based Q-values and trajectory stitching.

    This is the main function that replaces the original build_preference.
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

    # Load explored trajectories with MC rewards
    explored_traj = cal_monte_carlo_reward(args)

    # Create preference builder
    preference_builder = GraphPreferenceBuilder(
        graph=graph,
        step_threshold=args.step_threshold,
        traj_threshold=args.traj_threshold,
        stitching_threshold=1.1,  # Path must be 10% better
    )

    # Build preference data
    pm_data = preference_builder.build_preference_data(
        explored_trajectories=explored_traj,
        expert_data=golden_raw,
        instruction_part_len=len(instruction_part[args.task]),
        icl_prompt_part_len=len(icl_prompt_part[args.task]),
        enable_stitching=args.enable_stitching,
    )

    # Filter out invalid entries
    valid_pm_data = []
    for item in pm_data:
        if item.get('chosen') and item.get('rejected'):
            # Remove internal fields
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
    Original MC-based preference construction (fallback).

    Same as original construct_preference_monte_carlo_webshop.py
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
            if args.task == "webshop":
                id = item['id']
            elif args.task == "alfworld":
                game_file = item['game_file']
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
                    "prompt": gpt_conversations[:len(instruction_part[args.task])+1],
                    "chosen": agent_conversations[len(icl_prompt_part[args.task])+1: -1],
                    "rejected": gpt_conversations[len(instruction_part[args.task])+1:],
                })
            elif gpt_reward > agent_final_reward + traj_threshold:
                lose += 1
                global_traj += 1
                pm_data.append({
                    "id": int(f"{id}{iteration}"),
                    "prompt": gpt_conversations[:len(instruction_part[args.task])+1],
                    "chosen": gpt_conversations[len(instruction_part[args.task])+1:],
                    "rejected": agent_conversations[len(icl_prompt_part[args.task])+1: -1],
                })
            else:
                tie += 1

    if args.local_traj:
        for item in explored_traj:
            if args.task == "webshop":
                id = item['id']
            elif args.task == "alfworld":
                game_file = item['game_file']
                id = game_file_to_id.get(game_file, item['id'])

            iteration = item['iteration']
            if iteration == 0:
                continue

            expert_key = f"{id}_{iteration}"
            if expert_key not in golden_raw:
                continue

            if args.step_action:
                agent_step_conversations = template_change(item['agent_step_conversations'])
                agent_step_reward = item['monte_carlo_step_reward']
                agent_final_reward = item['agent_final_reward']

                gpt_step_conversations = golden_raw[expert_key]['gpt_step_conversations']
                gpt_step_reward = golden_raw[expert_key].get('monte_carlo_step_reward', golden_raw[expert_key].get('gpt_step_reward', 0))
                gpt_final_reward = golden_raw[expert_key]['gpt_reward']

                gpt_length = len(gpt_step_conversations)
                agent_length = gpt_length - len(instruction_part[args.task]) + len(icl_prompt_part[args.task])

                if is_empty_conversations(agent_step_conversations) or is_empty_conversations(gpt_step_conversations):
                    continue

                if agent_final_reward > gpt_final_reward + traj_threshold and agent_step_reward > gpt_step_reward + step_monte_carlo_threshold:
                    win += 1
                    local_step_traj += 1
                    pm_data.append({
                        "id": int(f"{id}{iteration}"),
                        "prompt": gpt_step_conversations[:gpt_length-1],
                        "chosen": [agent_step_conversations[agent_length-1]],
                        "rejected": [gpt_step_conversations[gpt_length-1]],
                    })
                elif gpt_final_reward > agent_final_reward + traj_threshold and gpt_step_reward > agent_step_reward + step_monte_carlo_threshold:
                    lose += 1
                    local_step_traj += 1
                    pm_data.append({
                        "id": int(f"{id}{iteration}"),
                        "prompt": gpt_step_conversations[:gpt_length-1],
                        "chosen": [gpt_step_conversations[gpt_length-1]],
                        "rejected": [agent_step_conversations[agent_length-1]],
                    })
                else:
                    tie += 1
            else:
                agent_conversations = template_change(item['agent_conversations'])
                agent_step_reward = item['monte_carlo_step_reward']
                agent_final_reward = item['agent_final_reward']

                gpt_conversations = golden_raw[expert_key]['gpt_conversations']
                gpt_step_reward = golden_raw[expert_key].get('monte_carlo_step_reward', golden_raw[expert_key].get('gpt_step_reward', 0))
                gpt_final_reward = golden_raw[expert_key]['gpt_reward']
                gpt_step_conversations = golden_raw[expert_key]['gpt_step_conversations']

                gpt_length = len(gpt_step_conversations)
                agent_length = gpt_length - len(instruction_part[args.task]) + len(icl_prompt_part[args.task])

                if is_empty_conversations(agent_conversations) or is_empty_conversations(gpt_conversations):
                    continue

                if agent_final_reward > gpt_final_reward + traj_threshold and agent_step_reward > gpt_step_reward + step_monte_carlo_threshold:
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
    parser = argparse.ArgumentParser("Graph-based Preference Construction")

    # Original arguments
    parser.add_argument(
        "--task",
        type=str,
        default="webshop",
        help="task name",
    )
    parser.add_argument(
        "--golden_traj_path",
        type=str,
        default="data/webshop_sft.json",
        help="path of the golden trajectory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='test.json',
        help="output path of the trajectory preference dataset",
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        default="",
        help="path to explored trajectories",
    )
    parser.add_argument(
        "--step_action",
        action="store_true",
        help="use step action or entire trajectory",
    )
    parser.add_argument(
        "--global_traj",
        action="store_true",
        help="use global trajectory"
    )
    parser.add_argument(
        "--local_traj",
        action="store_true",
        help="use local trajectory"
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--step_threshold",
        type=float,
        default=0.1,
        help="Step-level Q-value difference threshold"
    )
    parser.add_argument(
        "--traj_threshold",
        type=float,
        default=0.05,
        help="Trajectory-level score difference threshold"
    )

    # Graph-IPR specific arguments
    parser.add_argument(
        "--graph_path",
        type=str,
        default="",
        help="Path to the state graph file"
    )
    parser.add_argument(
        "--enable_stitching",
        action="store_true",
        help="Enable trajectory stitching for super-expert performance"
    )
    parser.add_argument(
        "--use_graph",
        action="store_true",
        help="Use graph-based preference construction instead of MC"
    )

    args = parser.parse_args()

    if args.use_graph and args.graph_path:
        build_preference_with_graph(args)
    else:
        # Fall back to original MC-based method
        if args.global_traj or args.local_traj:
            build_preference_mc(args)
        else:
            print("Please specify --global_traj or --local_traj, or use --use_graph with --graph_path")


if __name__ == "__main__":
    main()
