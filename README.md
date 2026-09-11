# CVD Risk Prediction from Retinal Fundus Images

Predicting cardiovascular disease (CVD) risk — specifically carotid intima-media thickness (CIMT), a validated early marker of atherosclerosis — from retinal fundus photographs, using a CNN baseline, a novel patient-similarity Graph Attention Network (GAT), and a CNN+GAT ensemble.

## Overview

Retinal microvasculature shares developmental origin with systemic vasculature, making fundus images a viable non-invasive window into cardiovascular risk. This project:

1. Establishes a CNN (ResNet18) baseline for binary CIMT classification (normal vs. thickened)
2. Introduces a **patient similarity graph GAT** — a novel approach that models each patient as a node in a graph, connected to their k-nearest neighbors by combined image-embedding and clinical similarity, letting a Graph Attention Network reason over relationships between patients rather than treating each image in isolation
3. Combines both models into a weighted ensemble for improved performance and complementary signal capture

## Motivation

Existing literature on retinal-fundus CVD prediction (Poplin et al. 2018, Diaz-Pinto et al. 2022, Zhang et al. 2023) treats each fundus image independently via CNNs or transformers. Graph-based approaches for retinal CVD prediction do exist (e.g. RGAT, vessel-topology GNNs) but build graphs from **vessel anatomy**. This project instead builds a graph from **patient-to-patient similarity** (image + clinical features) — a distinct graph construction paradigm not covered in prior work, to our knowledge.

## Dataset

**[China-Fundus-CIMT](https://springernature.figshare.com/articles/dataset/High-resolution_fundus_images_for_ophthalmomics_and_early_cardiovascular_disease_prediction_China_Fundus_Carotid_Intima-Media_Thickness_dataset/27907056)** (Guo, Fu, Li et al., *Scientific Data*, 2025)
- 2,903 patients, ~5,806 bilateral fundus images (512×512)
- Label: CIMT thickened (≥0.9mm) vs. normal (<0.9mm)
- Metadata: age, gender, raw CIMT value, per-eye image filenames

Not included in this repo due to size/licensing (CC BY-NC-ND) — download separately from the link above and update the `DATA_DIR`/`JSON_PATH` config cells in each notebook.

## Repository Structure

```
├── notebooks/
│   ├── china_fundus_cimt_pipeline_final.ipynb   # CNN (ResNet18) baseline
│   └── patient_gat_pipeline_final.ipynb          # GAT + CNN-GAT ensemble
├── requirements.txt
├── .gitignore
└── README.md
```

## Method Summary

### 1. CNN Baseline
ResNet18 (ImageNet pretrained), fine-tuned for binary classification. Patient-level train/val/test split (70/15/15) to prevent left/right-eye leakage.

### 2. Patient Similarity Graph GAT
- Extracts 512-dim image embeddings from the trained CNN (classifier head removed)
- Builds a **mutual k-NN graph** (k=15) over patients, edges weighted by inverse distance in a combined feature space (60% image embedding similarity, 40% clinical similarity — age + gender)
- 2-layer Graph Attention Network with residual connections, batch normalization, and edge-weighted attention
- Class-imbalance-aware loss (softened inverse-frequency weighting)
- Decision threshold tuned on validation data under a **minimum sensitivity constraint (≥0.85)** — appropriate for a clinical screening context, where missing an at-risk patient is costlier than a false alarm

### 3. CNN + GAT Ensemble
Weighted average of CNN and GAT predicted probabilities (w_gat=0.45), with threshold selected via a systematic sweep on validation data.

## Results

| Model | Accuracy | AUC-ROC | Sensitivity | Specificity | F1-score |
|---|---|---|---|---|---|
| ResNet18 (CNN) | 0.7913 | 0.8430 | 0.8523 | 0.6445 | 0.8523 |
| GAT (standalone) | 0.7798–0.7959 | 0.8195–0.8509 | 0.83–0.93 | 0.50–0.83 | 0.80–0.87 |
| **CNN + GAT Ensemble (final)** | **0.8119** | **0.8430** | **0.9253** | 0.5391 | **0.8742** |

Metrics reported: Accuracy, AUC-ROC (with 95% bootstrap CI), Sensitivity, Specificity, PPV, NPV, F1-score, Youden's J Index, Balanced Accuracy — matching standard reporting conventions in the retinal-fundus CVD literature.

## Interpretability

The GAT's attention weights can be inspected per patient to identify which neighboring patients most influenced a given prediction — a step toward clinically explainable risk scoring, absent from standard CNN approaches.

## Setup

```bash
pip install -r requirements.txt
```

`torch_geometric` may require a kernel restart after installation — see the first cell of `patient_gat_pipeline_final.ipynb`.

## Usage

1. Download the China-Fundus-CIMT dataset and update `DATA_DIR`/`JSON_PATH` in both notebooks
2. Run `china_fundus_cimt_pipeline_final.ipynb` completely (trains CNN, saves checkpoint + splits + probabilities)
3. Run `patient_gat_pipeline_final.ipynb` completely (builds graph, trains GAT, runs ensemble)


## Citation

If using the China-Fundus-CIMT dataset, cite:
> Guo, Fu, Li et al. "High-resolution fundus images for ophthalmomics and early cardiovascular disease prediction: China Fundus-Carotid Intima-Media Thickness dataset." *Scientific Data*, 2025.
