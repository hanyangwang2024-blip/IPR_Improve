#!/bin/bash

# Graph-IPR Pipeline for ALFWorld - vLLM Version (Much Faster!)
# Usage: bash run_pipeline_alfworld_graph_vllm.sh

# Cleanup function for Ctrl+C
cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Caught interrupt signal, cleaning up..."

    # Kill worker processes from pid files
    if [ -f "${logs_path}/worker_pid.txt" ]; then
        kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null
        rm -f ${logs_path}/worker_pid.txt
    fi
    if [ -f "${logs_path}/eval_pid.txt" ]; then
        kill -9 $(cat ${logs_path}/eval_pid.txt) 2>/dev/null
        rm -f ${logs_path}/eval_pid.txt
    fi

    # Kill any processes on the vLLM worker ports
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

# Set trap for SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Don't set global CUDA_VISIBLE_DEVICES - let each worker set its own GPU
# export CUDA_VISIBLE_DEVICES=3,4,5
export VLLM_USE_V1=0

model_name=Llama-2-7b-hf
task=alfworld
worker_num=2
exp_name=graph_exp

node_num=3  # number of GPUs
sample_num_workers=3  # number of vllm workers (one per GPU)

save_dir=/home/jimchen/temp_hanyang/IPR/checkpoints_${task}/
save_path=/home/jimchen/temp_hanyang/IPR/experiments/${model_name}-${task}-graph-ipr-vllm/
logs_path=${save_path}logs

# Graph-IPR specific parameters
max_graph_nodes=5000
gamma=0.99
vi_iterations=15
step_threshold=0.05
traj_threshold=0.01

sft_model_name=${exp_name}${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using SFT checkpoint: ${save_dir}${sft_model_name}"

# Check if SFT checkpoint exists
if [ ! -d "${save_dir}${sft_model_name}" ]; then
    sft_model_name=expLlama-2-7b-hf-alfworld-sft-step-entire-monte-carlo-beta-0.1-lr3e-6
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph-IPR SFT not found, using original: ${save_dir}${sft_model_name}"
fi

if [ ! -d "${save_dir}${sft_model_name}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: SFT checkpoint not found"
    exit 1
fi

mkdir -p ${save_path}
mkdir -p ${logs_path}/

cur_model_name=${sft_model_name}
monte_carlo_explore_model_name=${cur_model_name}-monte-carlo-explore
gpu_list=(3 4 5)

# Initialize global graph path
global_graph_path=${save_path}global_state_graph.pkl

for i in {1..6}; do
    echo ""
    echo "======================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph-IPR Iteration ${i} (vLLM)"
    echo "======================================"

    # ==================== Part 3: Explore ====================
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 1/5] Starting Explore phase..."
    explore_model_name=${cur_model_name}-explore

    # Create symlinks for explore models
    for ((j=0;j<${sample_num_workers};j++)); do
        [ ! -d "${save_dir}${explore_model_name}-${j}" ] && ln -s ${save_dir}${cur_model_name} ${save_dir}${explore_model_name}-${j}
    done

    # Launch vLLM workers
    rm -f ${logs_path}/worker_pid.txt
    fs_worker_port=31012
    worker_idx=0
    for ((j=0;j<${sample_num_workers};j++)); do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - vLLM Worker ${j} on GPU ${gpu_list[$((j % node_num))]} port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=${gpu_list[$((worker_idx % node_num))]} python -u -m fastchat.serve.vllm_worker \
            --model-path ${save_dir}${explore_model_name}-${j} \
            --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} \
            --gpu-memory-utilization 0.75 \
            --no-register >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$((fs_worker_port+1))
        worker_idx=$((worker_idx+1))
        sleep 15
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 60s for vLLM workers to initialize..."
    sleep 60

    step_traj_save_path=${save_path}${explore_model_name}
    rm -rf ${step_traj_save_path}
    mkdir -p ${step_traj_save_path}

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${worker_num} generate_response workers..."
    rm -f ${logs_path}/eval_pid.txt
    for (( j = 0; j <= $worker_num; j++ )); do
        python3 generate_response.py --exp_config ${task} --model_name ${explore_model_name}-$((j%sample_num_workers)) --part_num $((worker_num+1)) --part_idx ${j} --save_path ${step_traj_save_path} >> ${logs_path}/gen_response_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/eval_pid.txt
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for generate_response to complete..."
    wait $(cat ${logs_path}/eval_pid.txt)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generate response completed. Killing vLLM workers..."
    kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null
    rm -f ${logs_path}/worker_pid.txt
    sleep 5

    num_traj_files=$(ls ${step_traj_save_path}/*.json 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generated ${num_traj_files} trajectory files"

    # ==================== Part 4: Monte Carlo Sampling with Graph ====================
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 2/5] Starting Monte Carlo Sampling with Graph update..."

    # Create symlinks for MC explore models
    for ((j=0;j<${sample_num_workers};j++)); do
        [ ! -d "${save_dir}${monte_carlo_explore_model_name}-${j}" ] && ln -s ${save_dir}${sft_model_name} ${save_dir}${monte_carlo_explore_model_name}-${j}
    done

    # Launch vLLM workers for MC sampling
    rm -f ${logs_path}/worker_pid.txt
    fs_worker_port=31012
    worker_idx=0
    for ((j=0;j<${sample_num_workers};j++)); do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - vLLM Worker ${j} on GPU ${gpu_list[$((j % node_num))]} port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=${gpu_list[$((worker_idx % node_num))]} python -u -m fastchat.serve.vllm_worker \
            --model-path ${save_dir}${monte_carlo_explore_model_name}-${j} \
            --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} \
            --gpu-memory-utilization 0.75 \
            --no-register >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$((fs_worker_port+1))
        worker_idx=$((worker_idx+1))
        sleep 15
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 60s for vLLM workers to initialize..."
    sleep 60

    sample_workers=16
    monte_carlo_sample_save_path=${save_path}monte_carlo_sample_iteration_${i}/sampled_traj_0
    mkdir -p ${monte_carlo_sample_save_path}

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${sample_workers} Graph-IPR MC sampling workers..."
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - gamma=${gamma}, max_nodes=${max_graph_nodes}, vi_iter=${vi_iterations}"
    rm -f ${logs_path}/eval_pid.txt
    for ((l=0;l<$sample_workers; l++)); do
        python monte_carlo_sample_alfworld_graph.py \
            --agent_config fastchat_explore \
            --model_name ${monte_carlo_explore_model_name}-$((l%sample_num_workers)) \
            --exp_config ${task} \
            --part_num ${sample_workers} \
            --part_idx ${l} \
            --save_path ${monte_carlo_sample_save_path}/ \
            --data_path ${step_traj_save_path} \
            --gamma ${gamma} \
            --max_graph_nodes ${max_graph_nodes} \
            --vi_iterations ${vi_iterations} \
            >> ${logs_path}/mc_worker-${l}.log 2>&1 &
        echo $! >> ${logs_path}/eval_pid.txt
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for MC sampling to complete..."
    wait $(cat ${logs_path}/eval_pid.txt)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] MC sampling completed. Killing vLLM workers..."
    kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null
    rm -f ${logs_path}/worker_pid.txt
    sleep 5

    num_mc_files=$(ls ${monte_carlo_sample_save_path}/*.json 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generated ${num_mc_files} MC sample files"

    # ==================== Part 4.5: Merge Graphs ====================
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 3/5] Merging graphs from parallel workers..."
    python -m graph_ipr.merge_graphs \
        --input_dir ${monte_carlo_sample_save_path} \
        --output_path ${global_graph_path} \
        --max_nodes ${max_graph_nodes}

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph merge completed."

    # ==================== Part 5: Build Preference Data ====================
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 4/5] Building preference data using graph..."
    pm_data_path=${save_path}data_pm/${task}_${exp_name}_pm_${i}.json
    mkdir -p ${save_path}data_pm

    python construct_preference_graph_alfworld.py \
        --task $task \
        --output_path ${pm_data_path} \
        --traj_path ${step_traj_save_path} \
        --sample_path ${save_path}monte_carlo_sample_iteration_${i} \
        --graph_path ${global_graph_path} \
        --use_graph \
        --enable_stitching \
        --global_traj --local_traj \
        --traj_threshold ${traj_threshold} \
        --step_threshold ${step_threshold}

    # Check preference data count
    if [ -f "${pm_data_path}" ]; then
        pm_count=$(python -c "import json; print(len(json.load(open('${pm_data_path}'))))" 2>/dev/null || echo "0")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generated ${pm_count} preference pairs"
    else
        pm_count=0
    fi

    # Fallback to MC if no preference data
    if [ "${pm_count}" == "0" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Falling back to MC-based method..."
        python construct_preference_monte_carlo_${task}.py \
            --task $task \
            --output_path ${pm_data_path} \
            --traj_path ${step_traj_save_path} \
            --sample_path ${save_path}monte_carlo_sample_iteration_${i} \
            --global_traj --local_traj \
            --traj_threshold ${traj_threshold} \
            --step_threshold ${step_threshold}

        pm_count=$(python -c "import json; print(len(json.load(open('${pm_data_path}'))))" 2>/dev/null || echo "0")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] MC fallback generated ${pm_count} preference pairs"
    fi

    if [ "${pm_count}" == "0" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: No preference data, skipping DPO"
        continue
    fi

    # ==================== Part 6: DPO Training ====================
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 5/5] Starting DPO training..."
    dpo_model_name=${sft_model_name}-graph-dpo-iter-${i}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training: ${dpo_model_name} with ${pm_count} pairs"

    python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20002 fastchat/train/train_dpo.py \
        --model_name_or_path ${save_dir}${cur_model_name} \
        --ref_model_name_or_path ${save_dir}${sft_model_name} \
        --data_path ${pm_data_path} \
        --bf16 True \
        --output_dir ${save_dir}${dpo_model_name} \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --eval_strategy "no" \
        --save_strategy "no" \
        --beta 0.1 \
        --learning_rate 3e-6 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 5 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 4096 \
        --max_prompt_length 512 \
        --max_target_length 3072 \
        --gradient_checkpointing True

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DPO training completed"

    # ==================== Part 7: Evaluation ====================
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting evaluation of ${dpo_model_name}..."
    fs_worker_port=31012
    CUDA_VISIBLE_DEVICES=${gpu_list[0]} python -u -m fastchat.serve.vllm_worker \
        --model-path ${save_dir}${dpo_model_name} \
        --port ${fs_worker_port} \
        --worker-address http://localhost:${fs_worker_port} \
        --gpu-memory-utilization 0.75 \
        --no-register >> ${logs_path}/model_worker.log 2>&1 &
    fs_worker_pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 30s for eval worker to initialize..."
    sleep 30

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running evaluation..."
    python -m eval_agent.main --agent_config fastchat --model_name ${dpo_model_name} --exp_config ${task} --split test

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluation completed. Killing eval worker..."
    kill -9 $fs_worker_pid 2>/dev/null
    sleep 5

    cur_model_name=${dpo_model_name}

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iteration ${i} completed!"
done

echo ""
echo "======================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph-IPR Pipeline (vLLM) Completed!"
echo "======================================"
