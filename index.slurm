#!/bin/bash
#SBATCH --job-name=vit-train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G

# ---------------- ENVIRONMENT SETUP ----------------
ENV_DIR="$HOME/mtp/python-env-1080ti"

module purge
module load anaconda/2023.03-1
module load cuda/11.4

# Activate virtual environment
source $ENV_DIR/bin/activate
echo "[INFO] Activated virtual environment at $ENV_DIR"

# ---------------- JOB INFO ----------------
echo "======================================"
echo "[INFO] Node: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Start time: $(date)"
echo "======================================"

# ---------------- GPU CHECK ----------------
echo "[INFO] All GPUs on node (nvidia-smi):"
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "[INFO] GPU seen by PyTorch:"
python - <<'PYCODE'
import torch
if torch.cuda.is_available():
    print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        print(f"   Memory Reserved : {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
else:
    print("❌ CUDA not available in PyTorch runtime")
PYCODE

# ---------------- TRAINING SCRIPT ----------------
# python -u model-timesformer-person-train.py
python -u model-train-yolo.py
# python -u model-test-yolo.py
# python -u model-timesformer-person-test.py



# ---------------- JOB COMPLETION ----------------
echo "======================================"
echo "[INFO] Job finished at $(date)"
echo "======================================"
