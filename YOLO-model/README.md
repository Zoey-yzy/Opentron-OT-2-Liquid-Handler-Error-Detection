# YOLO-NAS Pipette Tip & Liquid Detection

This work builds on the computer vision approach introduced by Khan et al. (2025) for real-time quality control on the Opentrons OT-2. We fine-tune their YOLO-NAS-L pipeline on single-channel pipette images to detect tip presence and estimate aspirated volume.

Fine-tunes a **YOLO-NAS-L** model (starting from COCO pretrained weights) to detect two classes in OT-2 pipette images:

- **Tip** — the full transparent tip body
- **Liquid** — the colored liquid column inside the tip

Volume is estimated from the detection results:
```
volume_ul = (liquid_h / tip_h) × max_volume_ul
```

---

## Files

| File | Description |
|---|---|
| `prepare_labels.py` | Generates YOLO labels from captured images and builds the `dataset/` folder |
| `train_yolo.py` | Fine-tunes YOLO-NAS-L on the prepared dataset |
| `detect_tips.py` | Runs inference on a single image and outputs bounding boxes + volume estimate |
| `pipeline.py` | Live OT-2 pipeline — aspirate → detect → error check → dispense or stop |
| `captured_samples-selected.zip` | Raw training images captured from the OT-2 camera |
| `dataset.zip` | Pre-built labeled dataset (train/valid/test split, YOLO format) |

---

## Workflow

### Step 1 — Prepare the dataset

Unzip the raw images:
```bash
unzip captured_samples-selected.zip
```

Then generate YOLO label files and split into train/valid/test:
```bash
python prepare_labels.py
```

This reads images from `captured_samples-selected/`, computes bounding boxes from volume-fitted parameters, and writes the labeled dataset to `dataset/train/`, `dataset/valid/`, `dataset/test/`.

> Alternatively, unzip the pre-built dataset directly:
> ```bash
> unzip dataset.zip
> ```

---

### Step 2 — Fine-tune YOLO-NAS
Pre-trained model: https://drive.google.com/file/d/1XPc4e1gyOBGoK1aCCS8Wu2kcM8kH-QLr/view
Start from COCO pretrained weights:
```bash
python train_yolo.py
```

Common options:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `50` | Number of training epochs |
| `--batch` | `8` | Batch size (use `4` if memory errors on CPU) |
| `--data` | `./dataset` | Path to dataset folder from Step 1 |
| `--finetune` | — | Path to a `.pth` checkpoint to continue fine-tuning from |
| `--device` | auto | `cpu`, `mps` (Apple M1/M2), or `cuda` (NVIDIA) |

Continue fine-tuning from a previous checkpoint:
```bash
python train_yolo.py --finetune checkpoints/single_channel_tips/RUN_.../ckpt_best.pth --epochs 30
```

The best checkpoint is saved to:
```
checkpoints/single_channel_tips/RUN_<timestamp>/ckpt_best.pth
```

---

### Step 3 — Run inference

```bash
python detect_tips.py --image your_image.jpg --model checkpoints/.../ckpt_best.pth
```

Output:
```
  pipette_x            0.5059        normalized x center
  pipette_y            0.5791        normalized y center
  pipette_w            0.0620        normalized width
  pipette_h            0.6250        normalized height
  liquid_x             0.5019
  liquid_y             0.7832
  liquid_w             0.0449
  liquid_h             0.2881
  volume_ul           300.00  µL  (liquid_h/tip_h × 400)
```

An annotated image is saved alongside the input with tip (green) and liquid (orange) bounding boxes.

---

### Step 4 — Live demo on OT-2

```bash
python pipeline.py --robot-ip 169.254.122.0 --model checkpoints/.../ckpt_best.pth
```

For each well the pipeline:
1. Commands the OT-2 to aspirate and move to the camera position
2. Captures a frame and runs YOLO detection
3. Estimates volume and checks against the expected protocol volume (±15% tolerance)
4. **No error** → dispenses into the plate and continues
5. **Error** → returns liquid to reservoir, drops tip, logs the result, stops

---

## Requirements

```bash
pip install super-gradients torch opencv-python numpy requests
```

- Python 3.9+
- No GPU required (runs on CPU / Apple MPS)

---

## Training Details

| Setting | Value |
|---|---|
| Base model | YOLO-NAS-L, COCO pretrained |
| Classes | `Tip`, `Liquid` |
| Optimizer | AdamW, lr = 5×10⁻⁴, cosine decay |
| Epochs | 50 (early stop on mAP@0.50) |
| Metric | mAP@0.50 |
| Dataset size | ~85 train / 12 valid / 6 test images |
| Augmentation | Random affine ±5° rotation, standard YOLO augmentations |

---

## Reference

This project is based on the following paper:

> Khan, S., Møller, V., Frandsen, R., & Mansourvar, M. (2025). Real-time AI-driven quality control for laboratory automation: a novel computer vision solution for the Opentrons OT-2 liquid handling robot. *Applied Intelligence*, 55. https://doi.org/10.1007/s10489-025-06334-3
