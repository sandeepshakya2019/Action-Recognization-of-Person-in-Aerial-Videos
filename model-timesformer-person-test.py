# ============================================================
# evaluate_person_timesformer.py
# ============================================================
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from decord import VideoReader, cpu
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from torch.utils.data import Dataset, DataLoader
from transformers import TimesformerForVideoClassification, AutoImageProcessor

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "/home1/jtt_1/mtp/timesformer-person-dataset-test"
TEST_CSV = os.path.join(BASE_DIR, "test_person_labels.csv")

MODEL_PATH = "/home1/jtt_1/mtp/trained-timesformer-person/best_timesformer_epoch1.pt"
PRETRAIN_PATH = "/home1/jtt_1/mtp/timesformer_base_patch16_224_k400"

SAVE_DIR = "/home1/jtt_1/mtp/timesformer-eval-results"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_FRAMES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5

print(f"🚀 Evaluating on {DEVICE}")

# ============================================================
# DATASET CLASS
# ============================================================
class PersonVideoDataset(Dataset):
    def __init__(self, csv_file, processor, label_encoder, num_frames=8):
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.label_encoder = label_encoder
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = str(row["VideoPath"]).strip()
        if not os.path.isabs(video_path):
            video_path = os.path.join(BASE_DIR, video_path)
        video_path = os.path.normpath(video_path)

        actions = [a.strip() for a in str(row["Actions"]).split(",") if a.strip()]
        labels = self.label_encoder.transform([actions])[0]

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = [vr[i].asnumpy() for i in indices]
            inputs = self.processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        except Exception as e:
            print(f"⚠️ Error reading {video_path}: {e}")
            pixel_values = torch.zeros((3, self.num_frames, 224, 224))

        return pixel_values, torch.tensor(labels, dtype=torch.float32)

# ============================================================
# LOAD LABELS + MODEL
# ============================================================
test_df = pd.read_csv(TEST_CSV)

def extract_actions(df):
    all_actions = []
    for a in df["Actions"]:
        actions = [x.strip() for x in str(a).split(",") if x.strip()]
        all_actions.extend(actions)
    return sorted(list(set(all_actions)))

actions_list = extract_actions(test_df)
mlb = MultiLabelBinarizer()
mlb.fit([actions_list])
num_classes = len(actions_list)

print(f"🎯 Found {num_classes} unique actions: {actions_list}")

processor = AutoImageProcessor.from_pretrained(PRETRAIN_PATH)
model = TimesformerForVideoClassification.from_pretrained(PRETRAIN_PATH)
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ============================================================
# DATA LOADER
# ============================================================
test_ds = PersonVideoDataset(TEST_CSV, processor, mlb, NUM_FRAMES)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

# ============================================================
# INFERENCE LOOP
# ============================================================
all_preds, all_targets, all_probs = [], [], []

print(f"\n🎞️ Running inference on {len(test_ds)} person clips...\n")
for pixel_values, labels in tqdm(test_loader):
    pixel_values = pixel_values.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        preds = (probs > THRESH).astype(int)

    all_preds.extend(preds)
    all_probs.extend(probs)
    all_targets.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_targets = np.array(all_targets)

# ============================================================
# METRICS
# ============================================================
print("\n📊 Evaluating performance metrics...\n")

prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="micro", zero_division=0)
macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
mAP = average_precision_score(all_targets, all_probs, average="macro")

print(f"✅ Micro Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print(f"✅ Macro Precision: {macro_prec:.4f} | Recall: {macro_rec:.4f} | F1: {macro_f1:.4f}")
print(f"✅ Mean Average Precision (mAP): {mAP:.4f}")

# ============================================================
# PER-CLASS REPORT
# ============================================================
per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
    all_targets, all_preds, average=None, zero_division=0
)
per_class_ap = average_precision_score(all_targets, all_probs, average=None)

report = pd.DataFrame({
    "Action": actions_list,
    "Precision": per_class_prec,
    "Recall": per_class_rec,
    "F1": per_class_f1,
    "AP": per_class_ap
})

csv_path = os.path.join(SAVE_DIR, "test_class_report.csv")
report.to_csv(csv_path, index=False)

print(f"\n📁 Per-class results saved to: {csv_path}")
print(f"💾 Results directory: {SAVE_DIR}")

print("\n🏁 Evaluation complete!")
