#!/bin/bash

# 小批量测试脚本 - 只处理少量样本验证流程
# Usage: bash test_mc_sample.sh

# 先清理可能存在的旧进程
echo "Cleaning up old processes..."
pkill -9 -f "vllm_worker" 2>/dev/null
pkill -9 -f "fastchat.serve.vllm_worker" 2>/dev/null
for port in 31012 31013 31014; do
    pid=$(lsof -t -i :${port} 2>/dev/null)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null
    fi
done
sleep 5

export VLLM_USE_V1=0

model_name=Llama-2-7b-hf
task=alfworld
exp_name=graph_exp

save_dir=/home/jimchen/temp_hanyang/IPR/checkpoints_${task}/
save_path=/home/jimchen/temp_hanyang/IPR/experiments/${model_name}-${task}-graph-ipr-vllm/
logs_path=${save_path}logs

# Graph-IPR specific parameters
max_graph_nodes=5000
gamma=0.99
alpha=0.1
vi_iterations=15

sft_model_name=${exp_name}${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6

# Check if SFT checkpoint exists
if [ ! -d "${save_dir}${sft_model_name}" ]; then
    sft_model_name=expLlama-2-7b-hf-alfworld-sft-step-entire-monte-carlo-beta-0.1-lr3e-6
fi

monte_carlo_explore_model_name=${sft_model_name}-monte-carlo-explore
gpu_list=(0 1 6)

# 数据路径 (使用已有的explore数据)
step_traj_save_path=${save_path}${exp_name}${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6-explore

# 检查数据是否存在
if [ ! -d "${step_traj_save_path}" ]; then
    echo "ERROR: Data path not found: ${step_traj_save_path}"
    exit 1
fi

echo "Data path: ${step_traj_save_path}"
echo "Number of JSON files: $(ls ${step_traj_save_path}/*.json 2>/dev/null | wc -l)"

# 测试保存路径
test_save_path=${save_path}test_mc_sample/
rm -rf ${test_save_path}
mkdir -p ${test_save_path}

echo ""
echo "========================================"
echo "Testing MC sampling with small batch"
echo "========================================"

# 启动 vLLM worker
echo "Starting vLLM worker on GPU ${gpu_list[0]}..."

# 创建符号链接
[ ! -d "${save_dir}${monte_carlo_explore_model_name}-0" ] && ln -s ${save_dir}${sft_model_name} ${save_dir}${monte_carlo_explore_model_name}-0

CUDA_VISIBLE_DEVICES=${gpu_list[0]} python -u -m fastchat.serve.vllm_worker \
    --model-path ${save_dir}${monte_carlo_explore_model_name}-0 \
    --port 31012 \
    --worker-address http://localhost:31012 \
    --gpu-memory-utilization 0.9 \
    --no-register > ${logs_path}/test_worker.log 2>&1 &

WORKER_PID=$!
echo "Started vLLM worker (PID: ${WORKER_PID})"
echo "Waiting 60s for worker to initialize..."
sleep 60

echo ""
echo "Running MC sampling test (part_num=100, part_idx=0 -> ~213 samples)..."
echo ""

# 使用 part_num=100 只处理约 1% 的数据 (21308/100 ≈ 213 samples)
python monte_carlo_sample_alfworld_graph.py \
    --agent_config fastchat_explore \
    --model_name ${monte_carlo_explore_model_name}-0 \
    --exp_config ${task} \
    --part_num 1000 \
    --part_idx 0 \
    --save_path ${test_save_path} \
    --data_path ${step_traj_save_path} \
    --gamma ${gamma} \
    --max_graph_nodes ${max_graph_nodes} \
    --vi_iterations ${vi_iterations}

echo ""
echo "========================================"
echo "Test completed!"
echo "========================================"
echo "Results saved to: ${test_save_path}"
echo ""

# 检查结果
if [ -f "${test_save_path}alfworld_traj_0.json" ]; then
    result_count=$(python -c "import json; print(len(json.load(open('${test_save_path}alfworld_traj_0.json'))))" 2>/dev/null || echo "0")
    echo "Generated ${result_count} trajectory results"
fi

if [ -f "${test_save_path}state_graph.pkl" ]; then
    echo "State graph saved successfully"
fi

if [ -f "${test_save_path}graph_stats.json" ]; then
    echo "Graph stats:"
    cat ${test_save_path}graph_stats.json
fi

# 清理 worker
echo ""
echo "Cleaning up worker..."
kill -9 $WORKER_PID 2>/dev/null
pkill -9 -f "vllm_worker" 2>/dev/null
