#!/bin/bash

# Monte Carlo Sampling with Graph - Standalone Script
# Step 2 of Graph-IPR Pipeline
# Usage: bash run_mc_with_graph.sh [iteration_number]

set -e

# Cleanup function for Ctrl+C
cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Caught interrupt signal, cleaning up..."

    if [ -f "${logs_path}/worker_pid.txt" ]; then
        kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null
        rm -f ${logs_path}/worker_pid.txt
    fi
    if [ -f "${logs_path}/eval_pid.txt" ]; then
        kill -9 $(cat ${logs_path}/eval_pid.txt) 2>/dev/null
        rm -f ${logs_path}/eval_pid.txt
    fi

    for port in 31012 31013 31014; do
        pid=$(lsof -t -i :${port} 2>/dev/null)
        if [ -n "$pid" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing process on port ${port} (PID: ${pid})"
            kill -9 $pid 2>/dev/null
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleanup complete."
    exit 1
}

trap cleanup SIGINT SIGTERM

export VLLM_USE_V1=0

# ==================== Configuration ====================
model_name=Llama-2-7b-hf
task=alfworld
exp_name=graph_exp

node_num=3
sample_num_workers=3

save_dir=/home/jimchen/temp_hanyang/IPR/checkpoints_${task}/
save_path=/home/jimchen/temp_hanyang/IPR/experiments/${model_name}-${task}-graph-ipr-vllm/
logs_path=${save_path}logs

# Graph-IPR specific parameters
max_graph_nodes=5000
gamma=0.99
alpha=0.1
vi_iterations=15

# Iteration number (default: 1)
iteration=${1:-1}

# SFT model name
sft_model_name=${exp_name}${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6

# Check if SFT checkpoint exists
if [ ! -d "${save_dir}${sft_model_name}" ]; then
    sft_model_name=expLlama-2-7b-hf-alfworld-sft-step-entire-monte-carlo-beta-0.1-lr3e-6
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph-IPR SFT not found, using original: ${save_dir}${sft_model_name}"
fi

if [ ! -d "${save_dir}${sft_model_name}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: SFT checkpoint not found at ${save_dir}${sft_model_name}"
    exit 1
fi

# Current model for exploration (can be customized)
cur_model_name=${sft_model_name}
monte_carlo_explore_model_name=${cur_model_name}-monte-carlo-explore

# GPU list
gpu_list=(0 1 6)

# Explore data path (from Step 1)
# 直接使用指定的 explore 数据路径
step_traj_save_path=/home/jimchen/temp_hanyang/IPR/experiments/Llama-2-7b-hf-alfworld-graph-ipr-vllm/graph_expLlama-2-7b-hf-alfworld-sft-step-entire-monte-carlo-beta-0.1-lr3e-6-explore

# ==================== Validation ====================
mkdir -p ${save_path}
mkdir -p ${logs_path}/

echo "======================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monte Carlo Sampling with Graph (Iteration ${iteration})"
echo "======================================"
echo ""
echo "Configuration:"
echo "  - SFT Model: ${sft_model_name}"
echo "  - MC Explore Model: ${monte_carlo_explore_model_name}"
echo "  - Trajectory Data: ${step_traj_save_path}"
echo "  - gamma: ${gamma}"
echo "  - max_graph_nodes: ${max_graph_nodes}"
echo "  - vi_iterations: ${vi_iterations}"
echo ""

# Check if trajectory data exists (from Step 1)
if [ ! -d "${step_traj_save_path}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Trajectory data not found at ${step_traj_save_path}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Please run Step 1 (Explore) first."
    exit 1
fi

num_traj_files=$(ls ${step_traj_save_path}/*.json 2>/dev/null | wc -l)
if [ "${num_traj_files}" -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: No trajectory files found in ${step_traj_save_path}"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found ${num_traj_files} trajectory files from Step 1"

# ==================== Create Model Symlinks ====================
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating model symlinks..."
for ((j=0;j<${sample_num_workers};j++)); do
    if [ ! -d "${save_dir}${monte_carlo_explore_model_name}-${j}" ]; then
        ln -s ${save_dir}${sft_model_name} ${save_dir}${monte_carlo_explore_model_name}-${j}
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - Created symlink: ${monte_carlo_explore_model_name}-${j}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - Symlink exists: ${monte_carlo_explore_model_name}-${j}"
    fi
done

# ==================== Launch vLLM Workers ====================
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching vLLM workers..."
rm -f ${logs_path}/worker_pid.txt
fs_worker_port=31012
worker_idx=0

for ((j=0;j<${sample_num_workers};j++)); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - vLLM Worker ${j} on GPU ${gpu_list[$((j % node_num))]} port ${fs_worker_port}"
    CUDA_VISIBLE_DEVICES=${gpu_list[$((worker_idx % node_num))]} python -u -m fastchat.serve.vllm_worker \
        --model-path ${save_dir}${monte_carlo_explore_model_name}-${j} \
        --port ${fs_worker_port} \
        --worker-address http://localhost:${fs_worker_port} \
        --gpu-memory-utilization 0.9 \
        --no-register >> ${logs_path}/model_worker-${j}.log 2>&1 &
    echo $! >> ${logs_path}/worker_pid.txt
    fs_worker_port=$((fs_worker_port+1))
    worker_idx=$((worker_idx+1))
    sleep 15
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 60s for vLLM workers to initialize..."
sleep 60

# ==================== Run Monte Carlo Sampling ====================
sample_workers=3
monte_carlo_sample_save_path=${save_path}monte_carlo_sample_iteration_${iteration}/sampled_traj_0
mkdir -p ${monte_carlo_sample_save_path}

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${sample_workers} Graph-IPR MC sampling workers..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - gamma=${gamma}, max_nodes=${max_graph_nodes}, vi_iter=${vi_iterations}"

rm -f ${logs_path}/eval_pid.txt
for ((l=0;l<$sample_workers; l++)); do
    worker_save_path=${monte_carlo_sample_save_path}/worker_${l}
    mkdir -p ${worker_save_path}
    python monte_carlo_sample_alfworld_graph.py \
        --agent_config fastchat_explore \
        --model_name ${monte_carlo_explore_model_name}-$((l%sample_num_workers)) \
        --exp_config ${task} \
        --part_num ${sample_workers} \
        --part_idx ${l} \
        --save_path ${worker_save_path}/ \
        --data_path ${step_traj_save_path} \
        --gamma ${gamma} \
        --max_graph_nodes ${max_graph_nodes} \
        --vi_iterations ${vi_iterations} \
        >> ${logs_path}/mc_worker-${l}.log 2>&1 &
    echo $! >> ${logs_path}/eval_pid.txt
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - MC Worker ${l} started (PID: $!), saving to ${worker_save_path}"
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for MC sampling to complete..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] You can monitor progress with: tail -f ${logs_path}/mc_worker-*.log"
wait $(cat ${logs_path}/eval_pid.txt)

# ==================== Cleanup ====================
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MC sampling completed. Killing vLLM workers..."
kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null
rm -f ${logs_path}/worker_pid.txt
sleep 5

# ==================== Summary ====================
num_mc_files=$(find ${monte_carlo_sample_save_path} -name "*.json" 2>/dev/null | wc -l)
echo ""
echo "======================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monte Carlo Sampling with Graph Completed!"
echo "======================================"
echo ""
echo "Results:"
echo "  - Generated ${num_mc_files} MC sample files"
echo "  - Output directory: ${monte_carlo_sample_save_path}"
echo ""
echo "Next step: Run Step 3 (Merge Graphs) with:"
echo "  python -m graph_ipr.merge_graphs \\"
echo "      --input_dir ${monte_carlo_sample_save_path} \\"
echo "      --output_path ${save_path}global_state_graph.pkl \\"
echo "      --max_nodes ${max_graph_nodes} \\"
echo "      --gamma ${gamma} \\"
echo "      --alpha ${alpha}"
