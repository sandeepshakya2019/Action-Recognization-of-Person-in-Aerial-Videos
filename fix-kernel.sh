#!/bin/bash
#SBATCH --job-name=fix-kernel
#SBATCH --output=logs/fix-kernel-%j.out
#SBATCH --error=logs/fix-kernel-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=shortq
#SBATCH --qos=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# ============================================================
# 🧠 FIX JUPYTER KERNEL VISIBILITY FOR MMACTION ENVIRONMENT
# ============================================================

# Environment setup
ENV_DIR="$HOME/mtp/python-env-1080ti"
KERNEL_NAME="mmaction-env"
DISPLAY_NAME="MMACTION (1080Ti)"

echo "==========================================="
echo "[INFO] ✅ Starting Jupyter kernel setup"
echo "[INFO] ENVIRONMENT DIR : $ENV_DIR"
echo "[INFO] KERNEL NAME     : $KERNEL_NAME"
echo "[INFO] DISPLAY NAME    : $DISPLAY_NAME"
echo "==========================================="

# Load modules (adapt to your cluster setup)
module purge
module load anaconda/2023.03-1
module load cuda/11.4

# Activate Python environment
source $ENV_DIR/bin/activate
echo "[INFO] 🧩 Activated environment: $(which python)"

# Ensure Jupyter + ipykernel are installed
pip install --upgrade pip
pip install jupyter ipykernel --quiet

# Remove any old kernel with same name
if jupyter kernelspec list | grep -q "$KERNEL_NAME"; then
    echo "[INFO] Removing existing kernel $KERNEL_NAME..."
    jupyter kernelspec remove $KERNEL_NAME -y
fi

# Register new kernel
python -m ipykernel install --user --name=$KERNEL_NAME --display-name "$DISPLAY_NAME"

# Verify kernel installation
echo
echo "[INFO] ✅ Installed Jupyter kernels:"
jupyter kernelspec list

# Show Jupyter data paths
echo
echo "[INFO] 🔍 Jupyter paths:"
jupyter --paths

echo
echo "[INFO] 🎯 Done! Open VS Code → Kernel Picker → Select '$DISPLAY_NAME'"
echo "==========================================="
