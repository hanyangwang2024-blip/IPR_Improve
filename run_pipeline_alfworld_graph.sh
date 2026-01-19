#!/bin/bash

# Graph-IPR Pipeline for ALFWorld
# Usage: bash run_pipeline_alfworld_graph.sh

export CUDA_VISIBLE_DEVICES=3,4,5

# Start FastChat controller in background
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting FastChat controller..."
python -m fastchat.serve.controller --host 0.0.0.0 --port 21001 > /tmp/fastchat_controller.log 2>&1 &
controller_pid=$!
sleep 10
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Controller started with PID: ${controller_pid}"

# Cleanup function to kill controller on exit
cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up..."
    kill -9 $controller_pid 2>/dev/null
    pkill -f "fastchat.serve.model_worker" 2>/dev/null
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Controller stopped"
}
trap cleanup EXIT

model_name=Llama-2-7b-hf
task=alfworld
worker_num=2
exp_name=graph_exp

node_num=3
sample_num_workers=3

save_dir=/home/jimchen/temp_hanyang/IPR/checkpoints_${task}/
save_path=/home/jimchen/temp_hanyang/IPR/experiments/${model_name}-${task}-graph-ipr/
logs_path=${save_path}logs

# Graph-IPR specific parameters
max_graph_nodes=5000
gamma=0.99
vi_iterations=15
step_threshold=0.05
traj_threshold=0.01

sft_model_name=${exp_name}${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6

mkdir -p ${save_path}
mkdir -p ${save_dir}
mkdir -p ${logs_path}/

# ==================== Part 1: SFT Training (SKIPPED) ====================
echo ""
echo "======================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPING SFT Training - Using existing checkpoint"
echo "======================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using checkpoint: ${save_dir}${sft_model_name}"

# Verify checkpoint exists
if [ ! -d "${save_dir}${sft_model_name}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Checkpoint not found at ${save_dir}${sft_model_name}"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint verified"

# ==================== Part 2: Evaluate SFT Model (SKIPPED) ====================
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPING SFT evaluation - proceeding directly to IPR iterations"

cur_model_name=${sft_model_name}
monte_carlo_explore_model_name=${cur_model_name}-monte-carlo-explore
gpu_list=(3 4 5)

# Initialize global graph path
global_graph_path=${save_path}global_state_graph.pkl

for i in {1..6}; do
    echo ""
    echo "======================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph-IPR Iteration ${i}"
    echo "======================================"

    # ==================== Part 3: Explore ====================
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 1/5] Starting Explore phase..."
    explore_model_name=${cur_model_name}-explore

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating symlinks for explore models..."
    for ((j=0;j<${sample_num_workers};j++)); do
        [ ! -d "${save_dir}${explore_model_name}-${j}" ] && ln -s ${save_dir}${cur_model_name} ${save_dir}${explore_model_name}-${j}
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${sample_num_workers} model workers..."
    rm -f ${logs_path}/worker_pid.txt
    fs_worker_port=21012
    for ((j=0;j<${sample_num_workers};j++)); do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - Worker ${j} on GPU ${gpu_list[$((j % 3))]} port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=${gpu_list[$((j % 3))]} python -u -m fastchat.serve.model_worker \
            --model-path ${save_dir}${explore_model_name}-${j} --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$((fs_worker_port+1))
        sleep 10
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 30s for workers to initialize..."
    sleep 30

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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generate response completed. Killing model workers..."
    kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null
    sleep 5

    # Count generated files
    num_traj_files=$(ls ${step_traj_save_path}/*.json 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generated ${num_traj_files} trajectory files in ${step_traj_save_path}"

    # ==================== Part 4: Monte Carlo Sampling with Graph ====================
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 2/5] Starting Monte Carlo Sampling with Graph update..."

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating symlinks for MC explore models..."
    for ((j=0;j<${sample_num_workers};j++)); do
        [ ! -d "${save_dir}${monte_carlo_explore_model_name}-${j}" ] && ln -s ${save_dir}${sft_model_name} ${save_dir}${monte_carlo_explore_model_name}-${j}
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${sample_num_workers} model workers for MC sampling..."
    rm -f ${logs_path}/worker_pid.txt
    fs_worker_port=21012
    for ((j=0;j<${sample_num_workers};j++)); do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - Worker ${j} on GPU ${gpu_list[$((j % 3))]} port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=${gpu_list[$((j % 3))]} python -u -m fastchat.serve.model_worker \
            --model-path ${save_dir}${monte_carlo_explore_model_name}-${j} --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$((fs_worker_port+1))
        sleep 10
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 30s for workers to initialize..."
    sleep 30

    sample_workers=16  # 更多并行MC采样
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] MC sampling completed. Killing model workers..."
    kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null
    sleep 5

    # Count MC sample files
    num_mc_files=$(ls ${monte_carlo_sample_save_path}/*.json 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generated ${num_mc_files} MC sample files"

    # ==================== Part 4.5: Merge Graphs ====================
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 3/5] Merging graphs from parallel workers..."
    python -m graph_ipr.merge_graphs \
        --input_dir ${monte_carlo_sample_save_path} \
        --output_path ${global_graph_path} \
        --max_nodes ${max_graph_nodes}

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph merge completed. Stats:"
    if [ -f "${save_path}graph_stats.json" ]; then
        cat ${save_path}graph_stats.json
    else
        echo "  No stats file found"
    fi

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
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: No preference data file generated"
    fi

    # Fallback to MC if no preference data
    if [ "${pm_count}" == "0" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: No preference data, falling back to MC-based method..."
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

    # Skip DPO if still no data
    if [ "${pm_count}" == "0" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: No preference data available, skipping DPO training for this iteration"
        continue
    fi

    # ==================== Part 6: DPO Training ====================
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Step 5/5] Starting DPO training..."
    dpo_model_name=${sft_model_name}-graph-dpo-iter-${i}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training model: ${dpo_model_name}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using ${pm_count} preference pairs from ${pm_data_path}"

    python -m torch.distributed.run --nproc_per_node=3 --master_port=20002 fastchat/train/train_dpo.py \
        --model_name_or_path ${save_dir}${cur_model_name} \
        --ref_model_name_or_path ${save_dir}${sft_model_name} \
        --data_path ${pm_data_path} \
        --bf16 True \
        --output_dir ${save_dir}${dpo_model_name} \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
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
    fs_worker_port=21012
    CUDA_VISIBLE_DEVICES=3 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${dpo_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph-IPR Pipeline Completed!"
echo "Final graph statistics:"
cat ${save_path}graph_stats.json 2>/dev/null || echo "No stats file found"
echo "======================================"
