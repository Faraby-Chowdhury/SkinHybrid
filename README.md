# SkinHybrid: Hybrid FiLM-Conditioned Dermoscopic Lesion Classification
A hybrid ConvNeXt-Tiny model for 8-class dermoscopic lesion classification using FiLM-conditioned auxiliary branches and lesion-level splitting.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and evaluation pipeline for **SkinHybrid**: a hybrid classification architecture for 8-class dermoscopic lesion classification on ISIC 2019, incorporating patient metadata and lesion morphology via Feature-wise Linear Modulation (FiLM) with strict lesion-level group splitting.

---

## Key Highlights

- **Lesion-Level Splitting:** Strict grouping via lesion IDs and 64-bit DCT perceptual hashing ($Hamming \le 6$) across 25,331 images (13,311 groups), asserting **0% patient/lesion leakage** between splits.
- **FiLM Auxiliary Conditioning:** Metadata (age, sex, anatomical site) and morphology (16 handcrafted shape/color descriptors) modulate deep backbone features via scale and shift parameters initialized to identity ($\gamma=0, \beta=0$).
- **Principled Imbalance Correction:** Loss-level prior adjustment via **Balanced Softmax** replacing stacked focal/sampler combinations.
- **Quantitative Explainability:** Saliency maps evaluated using Attribution Mass Inside Lesion (AMIL), Deletion/Insertion Area Under Curve (AUC) against random controls, and weight-randomization sanity checks.

---

## Experimental Benchmark

Evaluation on the held-out test split of **4,222 images** (ISIC 2019):

| Model / Ensemble | Resolution | Accuracy | Balanced Acc. | Macro F1 | Macro AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ResNet50 (Plain Cross-Entropy) | 224px | 0.7532 | 0.5411 | 0.5688 | 0.9256 |
| ConvNeXt-Tiny + Balanced Softmax | 224px | 0.7527 | 0.6922 | 0.6132 | 0.9194 |
| Full Hybrid (Metadata + Morphology + 8x TTA) | 224px | 0.7672 | 0.7222 | 0.6368 | 0.9368 |
| **5-Model Ensemble (3x ConvNeXt-T + 2x EffNetV2-S)** | **384px** | **0.7949** | **0.7563** | **0.6772** | **0.9539** |
| *ISIC 2019 Challenge Winner (Official Sealed Set)* | *Ensemble* | — | *0.6360* | — | — |

---

## Project Structure

```text
├── SkinHybrid.ipynb      # End-to-end training, ablation, and XAI pipeline
├── download_data.py      # Automated dataset retrieval script
├── requirements.txt      # Dependency specification
├── SETUP.md              # Detailed environment setup instructions
└── figures/              # Qualitative attribution maps and ablation curves
