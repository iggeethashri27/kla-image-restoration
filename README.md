# KLA Image Restoration — SEMICON India Hackathon 2026

AI-based restoration of degraded semiconductor inspection images.  
Joint denoising (speckle + Gaussian noise) and 2× super-resolution in a single forward pass.

---

## Results

| Metric | Bicubic Baseline | Our Model |
|---|---|---|
| PSNR | ~27.5 dB | **29.15 dB** |
| Inference speed | — | **11.6 ms/image** (T4 GPU) |
| Output shape | 256×256 | 256×256 |
| Parameters | — | 4.95M |

---

## Repository Structure

```
kla-image-restoration/
  inference.py          ← evaluation script (run this)
  train.py              ← training script
  requirements.txt      ← dependencies
  src/
    model.py            ← NAFNet-based U-Net architecture
    dataset.py          ← data loading + synthetic degradation
    losses.py           ← Charbonnier + SSIM loss
    __init__.py
  runs/
    exp01/
      config.json       ← exact training config
      log.csv           ← per-epoch metrics
  results/
    restored_test/      ← restored outputs on test set (400 images)
  weights/
    best.pth            ← trained model weights (see download link below)
```

---

## Model Weights Download

The trained weights file (`best.pth`, ~20 MB) is hosted on Google Drive:

**[Download best.pth](YOUR_GOOGLE_DRIVE_LINK_HERE)**

Place it at `weights/best.pth` before running inference.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/kla-image-restoration.git
cd kla-image-restoration
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model weights
Download `best.pth` from the link above and place it in the `weights/` folder:
```bash
mkdir -p weights
# move downloaded file here
mv best.pth weights/best.pth
```

### 4. Run inference
```bash
python inference.py \
  --input_dir path/to/NoisyLR \
  --output_dir path/to/output \
  --weights weights/best.pth
```

No manual edits required. The script auto-detects GPU/CPU.

---

## Inference Script Usage

```
usage: inference.py [-h] --input_dir INPUT_DIR --output_dir OUTPUT_DIR
                    [--weights WEIGHTS] [--precision {fp32,fp16,bf16}]
                    [--batch_size BATCH_SIZE] [--device DEVICE]

arguments:
  --input_dir    Directory containing degraded input images (.npy format)
  --output_dir   Directory to save restored output images
  --weights      Path to trained model weights (default: weights/best.pth)
  --precision    Inference precision: fp32, fp16, bf16 (default: fp16)
  --batch_size   Batch size for inference (default: 16)
  --device       Device: cuda or cpu (default: cuda, auto-falls back to cpu)
```

### Example output
```
Restoring 400 images (batch_size=16)...
Reading images...
Done: 400 images in 4.64s (11.6 ms/img)
device=cuda  precision=fp16  batch_size=16  torch=2.10.0+cu128
```

---

## Training

### Dataset structure expected
```
data/
  GT/          ← clean ground-truth .npy images
  NoisyLR/     ← degraded low-resolution .npy images
```

### Run training
```bash
python train.py
```

Training config is in `runs/exp01/config.json`. Key hyperparameters:

| Parameter | Value |
|---|---|
| Architecture | NAFNet U-Net |
| Width | 32 channels |
| Encoder blocks | [2, 2, 4] |
| Middle blocks | 8 |
| Decoder blocks | [2, 2, 2] |
| Scale factor | 2× |
| Loss | Charbonnier (w=1.0) + SSIM (w=0.2) |
| Epochs | 150 |
| Batch size | 16 |
| LR | 2e-3 (cosine decay) |
| AMP | fp16 |
| EMA decay | 0.999 |

---

## Architecture

NAFNet-based U-Net with:
- LayerNorm2d normalization
- SimpleGate activation (replaces ReLU/GELU)
- Simplified Channel Attention
- PixelShuffle upsampling (no checkerboard artifacts)
- Bicubic global residual (zero-init head → starts at bicubic baseline)
- Reflect-padding for arbitrary input sizes

---

## Degradation Handling

The model handles all three KLA degradations in a single forward pass:
- Speckle noise (multiplicative)
- Additive Gaussian noise
- 2× downsampling

Training uses random-order synthetic degradation to improve generalization.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| GPU (training) | NVIDIA T4 (Kaggle) |
| GPU (eval target) | NVIDIA H100 |

---

## External Resources

| Resource | Link | License |
|---|---|---|
| NAFNet paper | [arxiv.org/abs/2204.04676](https://arxiv.org/abs/2204.04676) | — |
| PyTorch | [pytorch.org](https://pytorch.org) | BSD |
| lpips | [github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) | BSD |

---

## Team

- **Team name:** [YOUR TEAM NAME]
- **Institution:** [YOUR INSTITUTION]
- **Hackathon:** SEMICON India 2026 — KLA Problem Statement
