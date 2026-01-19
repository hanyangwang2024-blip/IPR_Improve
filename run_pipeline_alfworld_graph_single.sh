#!/bin/bash

# Graph-IPR Pipeline for ALFWorld - Single GPU (GPU 3)
# Usage: bash run_pipeline_alfworld_graph_single.sh

export CUDA_VISIBLE_DEVICES=3

# Start FastChat controller in background
echo "Starting FastChat controller..."
python -m fastchat.serve.controller --host 0.0.0.0 --port 21001 > /tmp/fastchat_controller.log 2>&1 &
controller_pid=$!
sleep 10
echo "Controller started with PID: ${controller_pid}"

# Cleanup function to kill controller on exit
cleanup() {
    echo "Cleaning up..."
    kill -9 $controller_pid 2>/dev/null
    echo "Controller stopped"
}
trap cleanup EXIT

model_name=Llama-2-7b-hf
task=alfworld
worker_num=8
exp_name=graph_exp

sample_num_workers=1

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

echo "Using SFT checkpoint: ${save_dir}${sft_model_name}"

# Check if SFT checkpoint exists, if not use the original SFT model
if [ ! -d "${save_dir}${sft_model_name}" ]; then
    sft_model_name=expLlama-2-7b-hf-alfworld-sft-step-entire-monte-carlo-beta-0.1-lr3e-6
    echo "Graph-IPR SFT not found, using original: ${save_dir}${sft_model_name}"
fi

if [ ! -d "${save_dir}${sft_model_name}" ]; then
    echo "ERROR: SFT checkpoint not found"
    exit 1
fi

mkdir -p ${save_path}
mkdir -p ${logs_path}/

cur_model_name=${sft_model_name}
monte_carlo_explore_model_name=${cur_model_name}-monte-carlo-explore

# Initialize global graph path
global_graph_path=${save_path}global_state_graph.pkl

for i in {1..6}; do
    echo "======================================"
    echo "Graph-IPR Iteration ${i}"
    echo "======================================"

    # Part 3: Explore
    explore_model_name=${cur_model_name}-explore
    [ ! -d "${save_dir}${explore_model_name}-0" ] && ln -s ${save_dir}${cur_model_name} ${save_dir}${explore_model_name}-0

    rm -f ${logs_path}/worker_pid.txt
    fs_worker_port=21012
    CUDA_VISIBLE_DEVICES=3 python -u -m fastchat.serve.model_worker \
        --model-path ${save_dir}${explore_model_name}-0 --port ${fs_worker_port} \
        --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-0.log 2>&1 &
    echo $! >> ${logs_path}/worker_pid.txt
    sleep 60

    step_traj_save_path=${save_path}${explore_model_name}
    rm -rf ${step_traj_save_path}
    mkdir -p ${step_traj_save_path}

    rm -f ${logs_path}/eval_pid.txt
    for (( j = 0; j <= $worker_num; j++ )); do
        python3 generate_response.py --exp_config ${task} --model_name ${explore_model_name}-0 --part_num $((worker_num+1)) --part_idx ${j} --save_path ${step_traj_save_path} >> ${logs_path}/gen_response_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/eval_pid.txt
    done
    wait $(cat ${logs_path}/eval_pid.txt)
    kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null

    # Part 4: Graph-IPR Monte Carlo sampling with graph update
    [ ! -d "${save_dir}${monte_carlo_explore_model_name}-0" ] && ln -s ${save_dir}${sft_model_name} ${save_dir}${monte_carlo_explore_model_name}-0

    rm -f ${logs_path}/worker_pid.txt
    fs_worker_port=21012
    CUDA_VISIBLE_DEVICES=3 python -u -m fastchat.serve.model_worker \
        --model-path ${save_dir}${monte_carlo_explore_model_name}-0 --port ${fs_worker_port} \
        --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-0.log 2>&1 &
    echo $! >> ${logs_path}/worker_pid.txt
    sleep 60

    sample_workers=4
    rm -f ${logs_path}/eval_pid.txt
    monte_carlo_sample_save_path=${save_path}monte_carlo_sample_iteration_${i}/sampled_traj_0
    mkdir -p ${monte_carlo_sample_save_path}

    # Use Graph-IPR sampling script (sequential on single GPU)
    for ((l=0;l<$sample_workers; l++)); do
        python monte_carlo_sample_alfworld_graph.py \
            --agent_config fastchat_explore \
            --model_name ${monte_carlo_explore_model_name}-0 \
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
    wait $(cat ${logs_path}/eval_pid.txt)
    kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null

    # Part 4.5: Merge graphs from parallel workers
    echo "Merging graphs from parallel workers..."
    python -m graph_ipr.merge_graphs \
        --input_dir ${monte_carlo_sample_save_path} \
        --output_path ${global_graph_path} \
        --max_nodes ${max_graph_nodes}
    echo "Graph merged."

    # Part 5: Build preference data using graph
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

    # Fallback to MC if no preference data
    if [ ! -s "${pm_data_path}" ] || [ "$(python -c "import json; print(len(json.load(open('${pm_data_path}'))))")" == "0" ]; then
        echo "Warning: No preference data, falling back to MC-based"
        python construct_preference_monte_carlo_${task}.py \
            --task $task \
            --output_path ${pm_data_path} \
            --traj_path ${step_traj_save_path} \
            --sample_path ${save_path}monte_carlo_sample_iteration_${i} \
            --global_traj --local_traj \
            --traj_threshold ${traj_threshold} \
            --step_threshold ${step_threshold}
    fi

    # Part 6: DPO Training (single GPU)
    dpo_model_name=${sft_model_name}-graph-dpo-iter-${i}
    CUDA_VISIBLE_DEVICES=3 python fastchat/train/train_dpo.py \
        --model_name_or_path ${save_dir}${cur_model_name} \
        --ref_model_name_or_path ${save_dir}${sft_model_name} \
        --data_path ${pm_data_path} \
        --bf16 True \
        --output_dir ${save_dir}${dpo_model_name} \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --eval_strategy "no" \
        --save_strategy "no" \
        --beta 0.1 \
        --learning_rate 3e-6 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 5 \
        --tf32 True \
        --model_max_length 4096 \
        --max_prompt_length 512 \
        --max_target_length 3072 \
        --gradient_checkpointing True

    # Part 7: Evaluate DPO model
    fs_worker_port=21012
    CUDA_VISIBLE_DEVICES=3 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${dpo_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &
    fs_worker_pid=$!
    sleep 60
    python -m eval_agent.main --agent_config fastchat --model_name ${dpo_model_name} --exp_config ${task} --split test
    kill -9 $fs_worker_pid 2>/dev/null

    cur_model_name=${dpo_model_name}
done

echo "======================================"
echo "Graph-IPR Pipeline Completed!"
echo "======================================"
