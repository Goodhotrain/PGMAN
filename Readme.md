# PGMAN: Cues to Semantics: Prompt-Guided Multimodal Alignment for Micro-video Emotion Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of **PGMAN**.

---

## 📢 News

- **Model Checkpoints:** Released model checkpoints for inference and evaluation, facilitating reproducibility and qualitative analysis.
- **Dataset Annotations:** Released processed annotation files for the ME-5 dataset (emotion-annotated MTSVRC) to support reproducibility and further research.
- **New Baselines:** Added experimental comparisons with recent methods.


---

## 🛠️ Preparation

### Model Weights

Our method relies on **VideoLLaMA2-7B**.  
Please download the pre-trained weights from the official HuggingFace page:

- https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B

---
## 🔓 Model Weights

To facilitate evaluation and qualitative analysis, we provide pre-trained model weights
of **PGMAN** for inference and demonstration purposes.

- The released checkpoint correspond to the models used in the revised experiments.
- These weights are intended for **inference and reproducibility of reported results**, rather than full training from scratch.
- Due to computational and licensing constraints of the backbone model, we do not release all intermediate training checkpoints.

[Download links (ME-5)](https://drive.google.com/file/d/1QqrIdo3CJcIMNXuMbE9Fu8_veRWQQ3DH/view?usp=sharing)
## 📁 Data Structure

Please organize your data directory (specified via `--root_path`) as follows:

```text
<root_path>/
├── MeiTu/
│   ├── video/              # Raw video files
│   └── audio/              # Extracted audio files
├── annotations/
│   ├── mtsvrc_title.json   # Caption or title annotations
│   └── mtsvrc_label.json   # Emotion labels
└── results/
    └── main/               # Logs and checkpoints

## Running Instructions

You can train and evaluate the model quickly with the following command:

```bash
$ python main.py
```

## 📊 Datasets

The experiments in this work are conducted on the following publicly available datasets:

- **Ekman-6 (Ek-6):**  
  https://drive.google.com/drive/folders/0B-iork9xj4brQmlYYjlsUUtVVGM

- **VideoEmotion-8 (EM-8):**  
  https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs

In addition, we provide the processed ME-5 **annotation files (JSON format)** used in our experiments via Google Drive:
- **Annotation files:** https://drive.google.com/file/d/1f7c2PZ6bOVTv0YP0yVsLcFEVt806nhbg/view?usp=drive_link

Please note that the raw videos of MTSVRC are subject to the original dataset license and must be obtained from the official source. Our released annotation files are intended to be used in conjunction with the officially downloaded data.

