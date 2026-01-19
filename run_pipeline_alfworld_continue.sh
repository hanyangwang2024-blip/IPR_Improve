#!/bin/bash

# IPR Pipeline for ALFWorld - Continue from SFT checkpoint
# Usage: bash run_pipeline_alfworld_continue.sh

export CUDA_VISIBLE_DEVICES=4,5

# Start FastChat controller in background
echo "Starting FastChat controller..."
python -m fastchat.serve.controller --host 0.0.0.0 --port 31001 > /tmp/fastchat_controller.log 2>&1 &
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
worker_num=1
exp_name=exp

node_num=2
sample_num_workers=2

save_dir=/home/jimchen/temp_hanyang/IPR/checkpoints_${task}/
save_path=/home/jimchen/temp_hanyang/IPR/experiments/${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6/
logs_path=${save_path}logs

sft_model_name=graph_exp${model_name}-${task}-sft-step-entire-monte-carlo-beta-0.1-lr3e-6

echo "Using SFT checkpoint: ${save_dir}${sft_model_name}"

if [ ! -d "${save_dir}${sft_model_name}" ]; then
    echo "ERROR: SFT checkpoint not found"
    exit 1
fi

mkdir -p ${save_path}
mkdir -p ${logs_path}/

cur_model_name=${sft_model_name}
monte_carlo_explore_model_name=${cur_model_name}-monte-carlo-explore
gpu_list=(4 5)

for i in {1..6}; do
    echo "======================================"
    echo "IPR Iteration ${i}"
    echo "======================================"

    # Part 3: Explore
    explore_model_name=${cur_model_name}-explore
    for ((j=0;j<${sample_num_workers};j++)); do
        [ ! -d "${save_dir}${explore_model_name}-${j}" ] && ln -s ${save_dir}${cur_model_name} ${save_dir}${explore_model_name}-${j}
    done

    rm -f ${logs_path}/worker_pid.txt
    fs_worker_port=31022
    for ((j=0;j<${sample_num_workers};j++)); do
        CUDA_VISIBLE_DEVICES=${gpu_list[$((j % 2))]} python -u -m fastchat.serve.model_worker \
            --model-path ${save_dir}${explore_model_name}-${j} --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$((fs_worker_port+1))
        sleep 15
    done
    sleep 60

    step_traj_save_path=${save_path}${explore_model_name}
    rm -rf ${step_traj_save_path}
    mkdir -p ${step_traj_save_path}

    rm -f ${logs_path}/eval_pid.txt
    for (( j = 0; j <= $worker_num; j++ )); do
        python3 generate_response.py --exp_config ${task} --model_name ${explore_model_name}-$((j%sample_num_workers)) --part_num $((worker_num+1)) --part_idx ${j} --save_path ${step_traj_save_path} >> ${logs_path}/gen_response_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/eval_pid.txt
    done
    wait $(cat ${logs_path}/eval_pid.txt)
    kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null

    # Part 4: Monte Carlo sampling
    for ((j=0;j<${sample_num_workers};j++)); do
        [ ! -d "${save_dir}${monte_carlo_explore_model_name}-${j}" ] && ln -s ${save_dir}${sft_model_name} ${save_dir}${monte_carlo_explore_model_name}-${j}
    done

    rm -f ${logs_path}/worker_pid.txt
    fs_worker_port=31022
    for ((j=0;j<${sample_num_workers};j++)); do
        CUDA_VISIBLE_DEVICES=${gpu_list[$((j % 2))]} python -u -m fastchat.serve.model_worker \
            --model-path ${save_dir}${monte_carlo_explore_model_name}-${j} --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
        echo $! >> ${logs_path}/worker_pid.txt
        fs_worker_port=$((fs_worker_port+1))
        sleep 15
    done
    sleep 60

    sample_workers=8
    rm -f ${logs_path}/eval_pid.txt
    monte_carlo_sample_save_path=${save_path}monte_carlo_sample_iteration_${i}/sampled_traj_0
    mkdir -p ${monte_carlo_sample_save_path}
    for ((l=0;l<$sample_workers; l++)); do
        python monte_carlo_sample_${task}.py --agent_config fastchat_explore --model_name ${monte_carlo_explore_model_name}-$((l%sample_num_workers)) --exp_config ${task} --part_num ${sample_workers} --part_idx ${l} --save_path ${monte_carlo_sample_save_path}/ --data_path ${step_traj_save_path} >> ${logs_path}/mc_worker-${l}.log 2>&1 &
        echo $! >> ${logs_path}/eval_pid.txt
    done
    wait $(cat ${logs_path}/eval_pid.txt)
    kill -9 $(cat ${logs_path}/worker_pid.txt) 2>/dev/null

    # Part 5: Build preference data
    pm_data_path=${save_path}data_pm/${task}_${exp_name}_pm_${i}.json
    mkdir -p ${save_path}data_pm
    python construct_preference_monte_carlo_${task}.py --task $task --output_path ${pm_data_path} --traj_path ${step_traj_save_path} --sample_path ${save_path}monte_carlo_sample_iteration_${i} --global_traj --local_traj --traj_threshold 0.01 --step_threshold 0.01

    # Part 6: DPO Training
    dpo_model_name=${sft_model_name}-dpo-iter-${i}
    python -m torch.distributed.run --nproc_per_node=2 --master_port=20002 fastchat/train/train_dpo.py \
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

    # Part 7: Evaluate DPO model
    fs_worker_port=31022
    CUDA_VISIBLE_DEVICES=4 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${dpo_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &
    fs_worker_pid=$!
    sleep 60
    python -m eval_agent.main --agent_config fastchat --model_name ${dpo_model_name} --exp_config ${task} --split test
    kill -9 $fs_worker_pid 2>/dev/null

    cur_model_name=${dpo_model_name}
done

echo "IPR Pipeline Completed!"
