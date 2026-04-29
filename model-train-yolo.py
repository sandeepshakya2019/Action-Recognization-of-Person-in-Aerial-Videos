from ultralytics import YOLO
import torch, gc, os

import torch
import gc
import os
import psutil

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

def free_gpu_memory(show_before_after: bool = True):
    """
    Frees up unused GPU and CPU memory (PyTorch + system level).

    Args:
        show_before_after (bool): If True, prints memory stats before/after cleanup.
    """
    if show_before_after:
        print("========= 🧠 Memory before cleanup =========")
        print_gpu_memory()
        print_system_memory()

    # ---- Clear PyTorch CUDA memory ----
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # ---- Run Python garbage collector ----
    gc.collect()

    if show_before_after:
        print("\n========= ✅ Memory after cleanup =========")
        print_gpu_memory()
        print_system_memory()
        print("===========================================")


def print_gpu_memory():
    """Helper to print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("GPU: No CUDA device available.")
        return
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"GPU: Allocated={gpu_mem:.2f} MB | Reserved={gpu_reserved:.2f} MB | Device={torch.cuda.get_device_name(0)}")


def print_system_memory():
    """Helper to print CPU (RAM) usage."""
    vm = psutil.virtual_memory()
    used_gb = vm.used / (1024 ** 3)
    total_gb = vm.total / (1024 ** 3)
    print(f"RAM: {used_gb:.2f} GB / {total_gb:.2f} GB ({vm.percent}%) used")

free_gpu_memory()

# ============================================================
# 🧩 CONFIGURATION
# ============================================================
DATA_CFG = "yolo-person.yaml"    # dataset config
MODEL_WEIGHTS = "yolov8m.pt"     # try 'yolov8n.pt' first, then 'yolov8s.pt' or 'yolov8m.pt' if GPU allows
SAVE_NAME = "person"
DEVICE = "cuda:0"                       # GPU index or 'cpu'
PROJECT_DIR = "yolo-model-run"

# ============================================================
# 🧠 MEMORY CLEANUP
# ============================================================
def clear_cuda_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    print("✅ GPU memory cleared.")

clear_cuda_memory()

# ============================================================
# 🚀 TRAINING
# ============================================================
model = YOLO(MODEL_WEIGHTS)

model.train(
    data=DATA_CFG,
    epochs=1,
    imgsz=640,
    batch=32,
    name=SAVE_NAME,
    device=DEVICE,

    # optimizer
    lr0=0.002,
    lrf=0.01,
    optimizer="AdamW",
    momentum=0.937,
    weight_decay=0.0005,

    # training behavior
    patience=10,
    cache=True,
    augment=True,
    project=PROJECT_DIR,
    exist_ok=True,
    verbose=True,

    # augmentations (tuned for aerial)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=10.0,
    translate=0.1,
    scale=0.4,
    shear=2.0,
    perspective=0.0003,

    flipud=0.1,
    fliplr=0.5,

    mosaic=0.7,
    mixup=0.1,

    label_smoothing=0.01,
    cos_lr=True,
    close_mosaic=10,
)

model.val(batch=1, workers=1)

print(f"🎯 Training complete! Check best weights in {PROJECT_DIR}/{SAVE_NAME}/weights/best.pt")
