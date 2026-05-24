# MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing

[![Paper](https://img.shields.io/badge/Paper-ACL_Review-blue)](main.tex)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

**MultiHaluDet** is a novel four-stage framework designed to detect multilingual hallucinations by probing the internal hidden state trajectories of frozen Large Language Models (LLMs). Unlike confidence-based heuristics, MultiHaluDet captures the "reasoning process" by analyzing distributional activation patterns across transformer layers.

## Key Framework Stages
1. **Dynamic Layer Probing:** Uniformly samples hidden states from any LLM architecture (Mistral, LLaMA).
2. **Multi-Scale Sequential Modeling:** Processes depth-wise features using a hybrid architecture of Multi-Scale Attention and Transformer Encoders.
3. **Out-of-Fold Feature Generation:** Produces unbiased deep embeddings via K-fold stacking to prevent data leakage.
4. **Log-Odds Ensemble Meta-Learning:** A learned logistic meta-regressor that weighs base classifiers (XGBoost, RF, SVM, etc.) in the log-odds space for robust, calibrated detection.

## Performance Summary
| Dataset | Base Model | AUROC (%) |
| :--- | :--- | :---: |
| **HaluEval** | Mistral-7B | 98.43 |
| **HaluEval** | LLaMA2-7B | 98.55 |
| **TriviaQA** | Mistral-7B | 98.30 |
| **TriviaQA** | LLaMA2-7B | 98.26 |

*MultiHaluDet consistently maintains strong performance across High (French), Medium (Bangla), and Low-resource (Amharic) languages without language-specific fine-tuning.*

## Installation

```bash
git clone https://github.com/alvi-uiu/MultiHaluDet.git
cd MultiHaluDet
pip install -r requirements.txt
```

## Usage

Run the 4-stage pipeline sequentially (feature extraction, deep training, and ensemble meta-learning):

```bash
# Detect hallucinations in HaluEval using Mistral-7B (runs all stages)
python run_pipeline.py --dataset halueval --model mistral-7b --lang en --stage all

# Or run stages individually:
python run_pipeline.py --dataset triviaqa --model llama2-7b --lang fr --stage extract
python run_pipeline.py --dataset triviaqa --model llama2-7b --lang fr --stage train_oof
python run_pipeline.py --dataset triviaqa --model llama2-7b --lang fr --stage ensemble
```

### Supported Models
The framework supports any model from the `src/config.py` registry:
- `mistral-7b`
- `llama2-7b`

## Repository Structure
```text
src/
├── config.py              # Hyperparameters & Model Registry
├── data/
│   ├── loader.py          # HaluEval & TriviaQA Data Loading
│   └── feature_extractor.py # LLM Probing & Statistical Feature Extraction
├── models/
│   ├── multihaludet.py    # MultiHaluDet Architecture
│   └── losses.py          # Focal, Asymmetric, & Contrastive Loss
├── training/
│   ├── trainer.py         # OOF training loops & EMA
│   └── augmentations.py   # MixUp & CutMix implementation
├── ensemble/
│   └── meta_learner.py    # Log-Odds Logistic Meta-Regressor
└── utils/
    ├── metrics.py         # Calibration & Evaluation (ECE, AUROC, Brier)
    └── visualization.py   # Plots
```

## Citation
If you find this code or methodology useful, please cite our paper:
```bibtex
@inproceedings{multihaludet2026,
  title={MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing},
  author={First Author and Second Author},
  booktitle={ACL},
  year={2026}
}
```
