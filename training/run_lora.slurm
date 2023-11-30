#!/bin/bash
#SBATCH --job-name=styledrop-lora-dreambooth-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/styledrop.out

set -x -e

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "START TIME: $(date)"

MUSE_REPO=/fsx/william/open-muse.2
OUTPUT_DIR=/fsx/william/styledrop-lora-dreambooth-2
LOG_PATH=$OUTPUT_DIR/main_log.txt
CONFIG_PATH=/fsx/william/amused/training/config_lora.yaml

mkdir -p $OUTPUT_DIR
touch $LOG_PATH
pushd $MUSE_REPO

CMD=" \
    /fsx/william/amused/training/training.py \
        --output_dir /fsx/william/styledrop-lora-dreambooth-2 \
        --mixed_precision fp16 \
        --report_to wandb
    "

GPUS_PER_NODE=1
NNODES=$SLURM_NNODES

# set the visible GPUs
export CUDA_VISIBLE_DEVICES=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens


# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH