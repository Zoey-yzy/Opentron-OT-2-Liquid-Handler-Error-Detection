"""
YOLO-NAS Training — Single-Channel Liquid Handler
==================================================
Follows the exact structure of tips_n_liquid.ipynb, adapted for:
  - 1-channel pipette (1 Tip, 1 Liquid per image)
  - macOS / CPU  (no CUDA required)
  - Dataset from targeted_pipette_data.csv via prepare_labels.py

Classes:  ["Tip", "Liquid"]   (same as original notebook)
Model:    yolo_nas_l           (same as original notebook)

After training, run detect_tips.py to get:
  pipette_x/y/w/h, liquid_x/y/w/h, estimated volume_ul

Run prepare_labels.py first to build the dataset folder.

Usage:
    python train_yolo.py
    python train_yolo.py --epochs 50 --batch 4
    python train_yolo.py --finetune ckpt_best.pth
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default=str(Path(__file__).parent / "dataset"),
    help="Dataset root produced by prepare_labels.py",
)
parser.add_argument("--epochs",  type=int,   default=50,
                    help="Training epochs (notebook used 100; 50 fine for small dataset)")
parser.add_argument("--batch",   type=int,   default=8,
                    help="Batch size (default 8 for M1 MPS; use 4 if memory errors)")
parser.add_argument("--workers", type=int,   default=0,
                    help="Dataloader workers (0 = safest on macOS)")
parser.add_argument(
    "--checkpoint_dir",
    default=str(Path(__file__).parent / "checkpoints"),
)
parser.add_argument("--experiment", default="single_channel_tips")
parser.add_argument(
    "--finetune",
    default=None,
    help="Path to .pth to fine-tune from (e.g. ckpt_best.pth). "
         "Omit to start from COCO pretrained weights.",
)
parser.add_argument(
    "--device",
    default=None,
    help="Device to train on: 'cpu', 'mps' (Apple M1/M2), 'cuda' (NVIDIA). "
         "Auto-detected if not set.",
)
args = parser.parse_args()

# ── Imports (mirrors notebook cell order) ────────────────────────────────────

import torch

# Compatibility fix: super_gradients uses torch.load without weights_only=False,
# which breaks on PyTorch 2.5+. Patch it before importing super_gradients.
_torch_load_orig = torch.load
torch.load = lambda *args, **kwargs: _torch_load_orig(
    *args, **{**kwargs, "weights_only": False}
)

from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)

# ── Config (mirrors notebook cell-2) ─────────────────────────────────────────
#
# Original notebook (8-channel):          This script (1-channel):
#   MODEL_ARCH = 'yolo_nas_l'        →      same
#   BATCH_SIZE = 16  (RTX 3090)      →      4  (CPU-friendly)
#   MAX_EPOCHS = 100                  →      50  (smaller dataset)
#   classes = ["Tip", "Liquid"]       →      same (still 2 classes)
#   num expected tips = 8             →      1

MODEL_ARCH      = "yolo_nas_l"
BATCH_SIZE      = args.batch
MAX_EPOCHS      = args.epochs
CHECKPOINT_DIR  = args.checkpoint_dir
EXPERIMENT_NAME = args.experiment
if args.device:
    DEVICE = args.device
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
CLASSES         = ["Tip", "Liquid"]   # unchanged from original

data_dir = Path(args.data)

print("=" * 60)
print("  YOLO-NAS | Single-Channel Liquid Handler")
print("=" * 60)
print(f"  Model      : {MODEL_ARCH}")
print(f"  Device     : {DEVICE}")
print(f"  Epochs     : {MAX_EPOCHS}  |  Batch: {BATCH_SIZE}")
print(f"  Classes    : {CLASSES}")
print(f"  Dataset    : {data_dir}")
print(f"  Checkpoints: {CHECKPOINT_DIR}/{EXPERIMENT_NAME}")
print(f"  Start from : {args.finetune if args.finetune else 'COCO pretrained'}")
print("=" * 60 + "\n")

# ── Trainer (mirrors notebook cell-3) ────────────────────────────────────────

trainer = Trainer(
    experiment_name=EXPERIMENT_NAME,
    ckpt_root_dir=CHECKPOINT_DIR,
)

# ── Dataset params (mirrors notebook cell-4) ─────────────────────────────────

dataset_params = {
    "data_dir"        : str(data_dir),
    "train_images_dir": "train/images",
    "train_labels_dir": "train/labels",
    "val_images_dir"  : "valid/images",
    "val_labels_dir"  : "valid/labels",
    "test_images_dir" : "test/images",
    "test_labels_dir" : "test/labels",
    "classes"         : CLASSES,
}

# ── Dataloaders (mirrors notebook cell-6) ────────────────────────────────────

train_data = coco_detection_yolo_format_train(
    dataset_params={
        "data_dir"  : dataset_params["data_dir"],
        "images_dir": dataset_params["train_images_dir"],
        "labels_dir": dataset_params["train_labels_dir"],
        "classes"   : dataset_params["classes"],
    },
    dataloader_params={
        "batch_size" : BATCH_SIZE,
        "num_workers": args.workers,
        "drop_last"  : False,
    },
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        "data_dir"  : dataset_params["data_dir"],
        "images_dir": dataset_params["val_images_dir"],
        "labels_dir": dataset_params["val_labels_dir"],
        "classes"   : dataset_params["classes"],
    },
    dataloader_params={
        "batch_size" : BATCH_SIZE,
        "num_workers": args.workers,
    },
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        "data_dir"  : dataset_params["data_dir"],
        "images_dir": dataset_params["test_images_dir"],
        "labels_dir": dataset_params["test_labels_dir"],
        "classes"   : dataset_params["classes"],
    },
    dataloader_params={
        "batch_size" : BATCH_SIZE,
        "num_workers": args.workers,
    },
)

# Augmentation tweak (mirrors notebook cell-10)
# Reduced rotation: 5° instead of 10.42° — single tip is always vertical
train_data.dataset.dataset_params["transforms"][1]["DetectionRandomAffine"]["degrees"] = 5.0

# ── Model (mirrors notebook cell-13) ─────────────────────────────────────────

if args.finetune:
    model = models.get(
        MODEL_ARCH,
        num_classes=len(CLASSES),
        checkpoint_path=args.finetune,
    )
    print(f"Loaded weights from: {args.finetune}\n")
else:
    model = models.get(
        MODEL_ARCH,
        num_classes=len(CLASSES),
        pretrained_weights="coco",
    )
    print("Loaded COCO pretrained weights.\n")

# ── Training params (mirrors notebook cell-16) ───────────────────────────────
#
# Key changes from original notebook:
#   mixed_precision: True  → False  (CUDA only; CPU will error otherwise)
#   optimizer: "Adam"      → "AdamW" (better weight decay handling)
#   ema decay: 0.9         → 0.9999  (standard EMA for object detection)
#   silent_mode: False     → False   (keep progress visible)

train_params = {
    "silent_mode"                      : False,
    "average_best_models"              : False,  # disabled — breaks on PyTorch 2.5+
    "warmup_mode"                      : "linear_epoch_step",
    "warmup_initial_lr"                : 1e-6,
    "lr_warmup_epochs"                 : 3,
    "initial_lr"                       : 5e-4,
    "lr_mode"                          : "cosine",
    "cosine_final_lr_ratio"            : 0.1,
    "optimizer"                        : "AdamW",
    "optimizer_params"                 : {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn" : True,
    "ema"                              : True,
    "ema_params"                       : {"decay": 0.9999, "decay_type": "threshold"},
    "max_epochs"                       : MAX_EPOCHS,
    "mixed_precision"                  : DEVICE == "cuda",   # CPU-safe
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(CLASSES),
        reg_max=16,
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(CLASSES),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
        )
    ],
    "metric_to_watch": "mAP@0.50",
}

# ── Train (mirrors notebook cell-17) ─────────────────────────────────────────

trainer.train(
    model=model,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data,
)

# ── Load best model and test (mirrors notebook cell-18, 19) ──────────────────

import os
# Find the run directory (super_gradients creates RUN_<timestamp> subfolder)
ckpt_root = Path(CHECKPOINT_DIR) / EXPERIMENT_NAME
run_dirs  = sorted(ckpt_root.glob("RUN_*"))
best_ckpt = run_dirs[-1] / "ckpt_best.pth" if run_dirs else ckpt_root / "ckpt_best.pth"

best_model = models.get(
    MODEL_ARCH,
    num_classes=len(CLASSES),
    checkpoint_path=str(best_ckpt),
)

print(f"\nLoaded best model from: {best_ckpt}")

# ── Test (mirrors notebook cell-19) ──────────────────────────────────────────

test_results = trainer.test(
    model=best_model,
    test_loader=test_data,
    test_metrics_list=DetectionMetrics_050(
        score_thres=0.1,
        top_k_predictions=300,
        num_cls=len(CLASSES),
        normalize_targets=True,
        post_prediction_callback=PPYoloEPostPredictionCallback(
            score_threshold=0.01,
            nms_top_k=1000,
            max_predictions=300,
            nms_threshold=0.7,
        ),
    ),
)

# ── Print results ─────────────────────────────────────────────────────────────
#
# Metrics explained (from notebook cell-20):
#   loss_cls  : how well the model classifies Tip vs Liquid
#   loss_iou  : how accurately bounding boxes are localized
#   loss_dfl  : precision of box edge prediction
#   Precision : of all predicted boxes, % that are correct
#   Recall    : of all real objects, % the model found
#   mAP@0.50  : mean Average Precision at IoU=0.5 (1.0 = perfect)
#   F1@0.50   : harmonic mean of Precision and Recall

print("\n" + "=" * 60)
print("  Test Results")
print("=" * 60)
for k, v in test_results.items():
    print(f"  {k:<35} {v:.4f}")
print("=" * 60)
print(f"\nBest checkpoint: {best_ckpt}")
print("Run detection:   python detect_tips.py --image your_image.jpg "
      f"--model {best_ckpt} --conf 0.7")
