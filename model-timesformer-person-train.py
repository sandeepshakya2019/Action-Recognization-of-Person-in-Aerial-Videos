# # ============================================================
# # train_g.py
# # ============================================================
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import TimesformerForVideoClassification, AutoImageProcessor, get_scheduler
# from torch.optim import AdamW

# from decord import VideoReader, cpu
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score
# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# import torch
# import gc

# def clear_cuda_memory():
#     """Safely clear all cached GPU memory and Python garbage."""
#     gc.collect()  # clear Python garbage
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()   # clear unused cached memory
#         torch.cuda.ipc_collect()   # release inter-process memory (optional)
#         print("🧹 CUDA memory cache cleared successfully!")
#     else:
#         print("⚠️ CUDA not available — nothing to clear.")

# # Example usage
# clear_cuda_memory()


# # ============================================================
# # CONFIGURATION
# # ============================================================
# BASE_DIR = "/home1/jtt_1/mtp/timesformer-person-dataset"
# TRAIN_CSV = os.path.join(BASE_DIR, "train_person_labels.csv")
# VAL_CSV = os.path.join(BASE_DIR, "val_person_labels.csv")

# MODEL_PATH = "/home1/jtt_1/mtp/timesformer_base_patch16_224_k400"
# SAVE_DIR = "/home1/jtt_1/mtp/trained-timesformer-person"
# os.makedirs(SAVE_DIR, exist_ok=True)

# NUM_FRAMES = 8
# EPOCHS = 1
# BATCH_SIZE = 16
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🚀 Training on: {DEVICE}")

# # ============================================================
# # DATASET CLASS
# # ============================================================
# class PersonVideoDataset(Dataset):
#     def __init__(self, csv_file, processor, label_encoder, num_frames=8):
#         self.df = pd.read_csv(csv_file)
#         self.processor = processor
#         self.label_encoder = label_encoder
#         self.num_frames = num_frames

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         video_path = str(row["VideoPath"]).strip()

#         if not os.path.isabs(video_path):
#             video_path = os.path.join(BASE_DIR, video_path)
#         video_path = os.path.normpath(video_path)

#         # Handle missing videos gracefully
#         if not os.path.exists(video_path):
#             print(f"⚠️ Missing: {video_path}")
#             dummy = torch.zeros((3, self.num_frames, 224, 224))
#             labels = torch.zeros(len(self.label_encoder.classes_))
#             return dummy, labels

#         # ---- Parse actions (multi-label) ----
#         actions = [a.strip() for a in str(row["Actions"]).split(",") if a.strip()]
#         labels = self.label_encoder.transform([actions])[0]

#         # ---- Load frames ----
#         try:
#             vr = VideoReader(video_path, ctx=cpu(0))
#             total_frames = len(vr)
#             indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
#             frames = [vr[i].asnumpy() for i in indices]
#         except Exception as e:
#             print(f"⚠️ Error reading {video_path}: {e}")
#             dummy = torch.zeros((3, self.num_frames, 224, 224))
#             labels = torch.zeros(len(self.label_encoder.classes_))
#             return dummy, labels

#         # ---- Preprocess for model ----
#         inputs = self.processor(frames, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].squeeze(0)
#         return pixel_values, torch.tensor(labels, dtype=torch.float32)

# # ============================================================
# # LOAD DATA + LABEL ENCODER
# # ============================================================
# train_df = pd.read_csv(TRAIN_CSV)
# val_df = pd.read_csv(VAL_CSV)

# # Collect all unique actions from dataset
# def extract_actions(df):
#     all_actions = []
#     for a in df["Actions"]:
#         actions = [x.strip() for x in str(a).split(",") if x.strip()]
#         all_actions.extend(actions)
#     return sorted(list(set(all_actions)))

# all_actions = extract_actions(train_df)
# label_encoder = MultiLabelBinarizer()
# label_encoder.fit([all_actions])
# num_classes = len(all_actions)

# print(f"🎯 Total action classes: {num_classes}")
# print(f"🧾 Actions: {all_actions}\n")

# # ============================================================
# # MODEL SETUP
# # ============================================================
# processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
# model = TimesformerForVideoClassification.from_pretrained(MODEL_PATH)

# # Replace classification head for our dataset
# model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
# model.to(DEVICE)

# # ============================================================
# # DATA LOADERS
# # ============================================================
# train_ds = PersonVideoDataset(TRAIN_CSV, processor, label_encoder, NUM_FRAMES)
# val_ds = PersonVideoDataset(VAL_CSV, processor, label_encoder, NUM_FRAMES)

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# # ============================================================
# # OPTIMIZER / SCHEDULER / LOSS
# # ============================================================
# optimizer = AdamW(model.parameters(), lr=LR)
# num_training_steps = EPOCHS * len(train_loader)
# lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
# criterion = torch.nn.BCEWithLogitsLoss()

# scaler = torch.amp.GradScaler('cuda')
# best_val_f1 = 0.0
# log_file = os.path.join(SAVE_DIR, "train_log.txt")

# # ============================================================
# # TRAINING LOOP
# # ============================================================
# for epoch in range(EPOCHS):
#     # ---------------- TRAIN ----------------
#     model.train()
#     train_loss = 0.0

#     for pixel_values, labels in tqdm(train_loader, desc=f"🟢 Epoch {epoch+1}/{EPOCHS} [Train]"):
#         pixel_values = pixel_values.to(DEVICE, non_blocking=True)
#         labels = labels.to(DEVICE, non_blocking=True)

#         with torch.amp.autocast('cuda'):
#             outputs = model(pixel_values)
#             loss = criterion(outputs.logits, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)
#         lr_scheduler.step()
#         train_loss += loss.item()

#     avg_train_loss = train_loss / len(train_loader)

#     # ---------------- VALIDATION ----------------
#     model.eval()
#     val_loss = 0.0
#     preds_all, targets_all = [], []

#     with torch.no_grad():
#         for pixel_values, labels in tqdm(val_loader, desc=f"🔵 Epoch {epoch+1}/{EPOCHS} [Val]"):
#             pixel_values = pixel_values.to(DEVICE)
#             labels = labels.to(DEVICE)

#             outputs = model(pixel_values)
#             loss = criterion(outputs.logits, labels)
#             val_loss += loss.item()

#             preds = (torch.sigmoid(outputs.logits) > 0.5).int().cpu().numpy()
#             preds_all.extend(preds)
#             targets_all.extend(labels.cpu().numpy())

#     avg_val_loss = val_loss / len(val_loader)
#     val_f1 = f1_score(targets_all, preds_all, average="micro")

#     print(f"\n📊 Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | F1: {val_f1:.4f}")

#     # Save best model
#     if val_f1 > best_val_f1:
#         best_val_f1 = val_f1
#         best_model_path = os.path.join(SAVE_DIR, f"best_timesformer_epoch{epoch+1}.pt")
#         torch.save(model.state_dict(), best_model_path)
#         print(f"✅ Saved best model → {best_model_path} (F1={val_f1:.4f})")

#     with open(log_file, "a") as f:
#         f.write(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_f1={val_f1:.4f}\n")

# # ============================================================
# # FINAL SUMMARY
# # ============================================================
# print(f"\n🏁 Training complete! Best F1-score: {best_val_f1:.4f}")
# print(f"💾 Models saved in: {SAVE_DIR}")
# print(f"🧾 Logs written to: {log_file}")


# ============================================================
# train_g.py (Overfitting-Protected + Graph Logging)
# ============================================================
import os
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import TimesformerForVideoClassification, AutoImageProcessor, get_scheduler
from decord import VideoReader, cpu
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CLEAR GPU MEMORY
# ============================================================
def clear_cuda_memory():
    """Safely clear cached GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("🧹 CUDA memory cache cleared.")
    else:
        print("⚠️ CUDA not available.")

clear_cuda_memory()

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "/home1/jtt_1/mtp/timesformer-person-dataset"
TRAIN_CSV = os.path.join(BASE_DIR, "train_person_labels.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_person_labels.csv")

MODEL_PATH = "/home1/jtt_1/mtp/timesformer_base_patch16_224_k400"
SAVE_DIR = "/home1/jtt_1/mtp/trained-timesformer-person"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_FRAMES = 8
EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Training on: {DEVICE}")

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

        if not os.path.exists(video_path):
            print(f"⚠️ Missing video: {video_path}")
            dummy = torch.zeros((3, self.num_frames, 224, 224))
            labels = torch.zeros(len(self.label_encoder.classes_))
            return dummy, labels

        actions = [a.strip() for a in str(row["Actions"]).split(",") if a.strip()]
        labels = self.label_encoder.transform([actions])[0]

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = [vr[i].asnumpy() for i in indices]
        except Exception as e:
            print(f"⚠️ Error reading {video_path}: {e}")
            dummy = torch.zeros((3, self.num_frames, 224, 224))
            labels = torch.zeros(len(self.label_encoder.classes_))
            return dummy, labels

        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(labels, dtype=torch.float32)

# ============================================================
# DATA & LABELS
# ============================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

def extract_actions(df):
    all_actions = []
    for a in df["Actions"]:
        actions = [x.strip() for x in str(a).split(",") if x.strip()]
        all_actions.extend(actions)
    return sorted(list(set(all_actions)))

all_actions = extract_actions(train_df)
label_encoder = MultiLabelBinarizer()
label_encoder.fit([all_actions])
num_classes = len(all_actions)

print(f"🎯 Classes: {num_classes}")
print(f"🧾 Actions: {all_actions}\n")

# ============================================================
# MODEL SETUP
# ============================================================
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = TimesformerForVideoClassification.from_pretrained(MODEL_PATH)

# Add dropout regularization (anti-overfitting)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.3

# Replace classifier head
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
model.to(DEVICE)

# ============================================================
# DATA LOADERS
# ============================================================
train_ds = PersonVideoDataset(TRAIN_CSV, processor, label_encoder, NUM_FRAMES)
val_ds = PersonVideoDataset(VAL_CSV, processor, label_encoder, NUM_FRAMES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# OPTIMIZER / SCHEDULER / LOSS
# ============================================================
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=int(0.1 * num_training_steps),
                             num_training_steps=num_training_steps)
criterion = torch.nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler('cuda')

# ============================================================
# TRAINING LOOP WITH TRACKING
# ============================================================
best_val_f1 = 0.0
no_improve_epochs = 0
log_file = os.path.join(SAVE_DIR, "train_log.txt")

# For plotting
train_losses, val_losses, val_f1_scores = [], [], []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for pixel_values, labels in tqdm(train_loader, desc=f"🟢 Epoch {epoch+1}/{EPOCHS} [Train]"):
        pixel_values = pixel_values.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0.0
    preds_all, targets_all = [], []

    with torch.no_grad():
        for pixel_values, labels in tqdm(val_loader, desc=f"🔵 Epoch {epoch+1}/{EPOCHS} [Val]"):
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs.logits) > 0.5).int().cpu().numpy()
            preds_all.extend(preds)
            targets_all.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_f1 = f1_score(targets_all, preds_all, average="micro")
    val_f1_scores.append(val_f1)

    print(f"\n📊 Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | F1: {val_f1:.4f}")

    # Early stopping & best model saving
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        no_improve_epochs = 0
        best_model_path = os.path.join(SAVE_DIR, f"best_timesformer_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved new best model → {best_model_path} (F1={val_f1:.4f})")
    else:
        no_improve_epochs += 1
        print(f"⚠️ No improvement for {no_improve_epochs} epochs.")

    if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
        print(f"🛑 Early stopping triggered after {no_improve_epochs} stagnant epochs.")
        break

    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_f1={val_f1:.4f}\n")

    clear_cuda_memory()

# ============================================================
# FINAL SUMMARY + PLOT
# ============================================================
print(f"\n🏁 Training complete! Best F1: {best_val_f1:.4f}")
print(f"💾 Models saved in: {SAVE_DIR}")
print(f"🧾 Logs: {log_file}")

# ---- Plot Loss & F1 ----
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
plt.plot(val_f1_scores, label='Validation F1-score', color='green', linewidth=2)

plt.title("Training Progress (Loss & F1-score)")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plot_path = os.path.join(SAVE_DIR, "training_curve.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"📈 Training curve saved → {plot_path}")
