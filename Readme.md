# PGMAN

**Cues to Semantics: Prompt-Guided Multimodal Alignment for Micro-video Emotion Recognition**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

PGMAN combines visual, audio, and caption cues for micro-video emotion
recognition. Video captions are generated offline with VideoLLaMA2-7B and
cached as JSON annotations, so training and cached-caption inference do not
need to keep the captioning model in memory.

## Method overview

The current pipeline contains four main parts:

1. a divided space-time transformer and a frame-level ViT for visual cues;
2. a Wav2Vec2-based audio encoder and BERT caption encoder;
3. **DecAlign**, which decomposes multimodal alignment into visual-text,
   visual-audio, and text-audio contrastive objectives;
4. cross-modal attention, bottleneck fusion, and an emotion classifier.

`models/decalign.py` contains the standalone DecAlign implementation. Pairwise
losses can be returned separately or reweighted through `pair_weights`, which
makes alignment ablations easier to reproduce.

## Repository layout

```text
.
├── main.py                 # Training entry point
├── opts.py                 # Command-line configuration
├── models/
│   ├── pgman.py            # Full PGMAN model
│   ├── decalign.py         # Decomposed multimodal alignment
│   ├── vit.py              # Visual encoders
│   ├── at2.py              # Audio encoder
│   └── mbt_fusion.py       # Bottleneck fusion
├── datasets/               # Dataset and DataLoader definitions
├── transforms/             # Video/audio preprocessing
├── core/                   # Loss, optimizer, and runtime helpers
├── train.py
└── validation.py
```

## Preparation

The caption annotations used by PGMAN are produced with
[VideoLLaMA2-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B). Caption
generation is an offline preprocessing step; `main.py` reads the cached JSON
file directly.

The expected data layout is:

```text
<root_path>/
├── MeiTu/
│   ├── video/
│   └── audio/
├── annotations/
│   ├── mtsvrc_title.json
│   ├── mtsvrc_label.json
│   └── mtsvrc.csv
└── results/
```

## Running

Show all options:

```bash
python main.py --help
```

Start the default ME-5 experiment:

```bash
python main.py \
  --root_path /path/to/data \
  --dataset ME5 \
  --n_classes 5
```

Resume from a checkpoint:

```bash
python main.py \
  --root_path /path/to/data \
  --pretrained /path/to/checkpoint.pth
```

The default setup samples 8 frames at 224 × 224 resolution. Results,
TensorBoard logs, and checkpoints are written below `results/main` inside the
specified root directory.

## Datasets

Experiments use the following public datasets:

- [Ekman-6](https://drive.google.com/drive/folders/0B-iork9xj4brQmlYYjlsUUtVVGM)
- [VideoEmotion-8](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs)
- ME-5 (emotion-annotated MTSVRC); processed annotations are available from
  the [annotation download](https://drive.google.com/file/d/1f7c2PZ6bOVTv0YP0yVsLcFEVt806nhbg/view?usp=drive_link)

Raw MTSVRC videos remain subject to the original dataset license and are not
redistributed by this repository.
