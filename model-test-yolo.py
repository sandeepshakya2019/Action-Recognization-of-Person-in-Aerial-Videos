# ============================================================
# 📊 YOLOv8 Evaluation (mAP, Precision, Recall)
# ============================================================
from ultralytics import YOLO
import torch
import os

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================
MODEL_PATH = "/home1/jtt_1/mtp/models-2/yolo-person-model.pt"  # your trained model
DATA_CFG = "yolo-person.yaml"                                # dataset config YAML
PROJECT_DIR = "runs-person/test"                              # where results will be stored
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_THRESH = 0.6
IOU_THRESH = 0.45
IMGSZ = 1280  # should match your training size

# ============================================================
# 🧠 LOAD MODEL
# ============================================================
print(f"[INFO] Loading model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f"[INFO] Model loaded on {DEVICE.upper()}")

# ============================================================
# 🎯 EVALUATE ON TEST SPLIT
# ============================================================
results = model.val(
    data=DATA_CFG,
    split="test",          # evaluates using test set from YAML
    imgsz=IMGSZ,
    conf=CONF_THRESH,
    iou=IOU_THRESH,
    save_json=True,        # save COCO-format results
    save_txt=True,         # save per-image detections
    project=PROJECT_DIR,
    name="yolo-person-test",
    exist_ok=True,
    device=DEVICE,
    verbose=True
)

# ============================================================
# 🧾 PRINT SUMMARY
# ============================================================
print("\n===================== 📈 Evaluation Summary =====================")
print(f"✅ mAP@50: {results.box.map50:.4f}")
print(f"✅ mAP@50-95: {results.box.map:.4f}")
print(f"✅ Precision: {results.box.p:.4f}")
print(f"✅ Recall:    {results.box.r:.4f}")
print(f"✅ F1-Score:  {results.box.f1:.4f}")
print("=================================================================")
print(f"\n📁 Detailed results saved in: {results.save_dir}")
