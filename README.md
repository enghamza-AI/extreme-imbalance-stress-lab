# ⚡ Extreme Imbalance Stress Lab

> 6 techniques. 1 dataset. 99:1 odds. Only one wins.

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/enghamza-AI/flux)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License: apache](https://img.shields.io/badge/License-Apache-yellow)](LICENSE)

## What Is This?

Most ML tutorials use balanced datasets. Real world data is never balanced.

Cybersecurity data is 99% normal traffic. 1% actual attacks.
Fraud data is 99.9% legitimate. 0.1% fraud.
Cancer screening is 95% healthy. 5% positive.

If you train a model naively on this — it learns one trick:
**predict the majority class every time.**
It gets 99% accuracy. It catches zero attacks.

This project stress-tests 6 imbalance handling techniques on the
DARPA KDD Cup 1999 intrusion detection dataset (99:1 imbalance ratio)
and measures each one across 4 dimensions:
F1 Score · Recall · Inference Time · Memory Usage

The result is an original **technique selection flowchart** —
given your dataset stats, which technique should you actually use?

## The 6 Techniques

| # | Technique | Strategy |
|---|-----------|----------|
| 1 | SMOTE | Creates synthetic minority samples |
| 2 | ADASYN | Adaptive synthetic — focuses on hard regions |
| 3 | ROS | Random oversampling — simple duplication |
| 4 | RUS | Random undersampling — delete majority |
| 5 | class_weight | No resampling — penalize wrong predictions |
| 6 | Balanced RF | Algorithm-level balancing per tree |

## Tournament Results

*(Results populate after running the tournament)*
CHECK OUTPUTS FOLDER

## The Selection Flowchart

*(Flowchart image will be added after project completion)*
CHECK OUTPUTS FOLDER 

## Live Demo

🚀 Try it live on HuggingFace Spaces:
[https://huggingface.co/spaces/enghamza-AI/flux](https://huggingface.co/spaces/enghamza-AI/flux)

Upload any binary classification CSV → get technique recommendation instantly.

## What I Learned

- `imbalanced-learn` — SMOTE, ADASYN, ROS, RUS, Balanced RF
- `tracemalloc` — memory profiling in Python
- Time benchmarking for ML pipelines
- Why F1 and Recall matter more than accuracy on skewed data
- Decision flowchart design from experiment results
- Cybersecurity dataset handling (KDD Cup 1999)

## Tech Stack

- Python 3.10+
- scikit-learn
- imbalanced-learn
- pandas + numpy
- matplotlib
- streamlit

## project structure

extreme-imbalance-stress-lab/

├── src/

│   ├── data_loader.py      # loads + cleans KDD dataset

│   ├── techniques.py       # all 6 balancing techniques

│   ├── evaluator.py        # F1, recall, time, memory tracking

│   ├── tournament.py       # runs the full bracket

│   └── flowchart.py        # builds selection flowchart

├── app/

│   └── streamlit_app.py    # HuggingFace Spaces dashboard

└── outputs/                # saved charts + results


## Author

**Hamza** — BSAI Student · AI Systems Engineering Self-Study
- GitHub: [github.com/enghamza-AI](https://github.com/enghamza-AI)
- HuggingFace: [huggingface.co/spaces/enghamza-AI](https://huggingface.co/spaces/enghamza-AI)

---

*Part of the Diamond AI Roadmap — Stage 2, Week 3*
- HuggingFace Spaces

## Project Structure
