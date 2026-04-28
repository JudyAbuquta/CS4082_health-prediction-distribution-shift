# Health Risk Prediction Under Dataset Shift
### CS4082: Machine Learning — Final Project | Effat University | Spring 2026

> **Adversarial Challenge:** Distribution Shift (Out-of-Distribution Generalization)

---

## Overview

Binary cardiac event classifiers are trained on the UCI Heart Disease dataset and evaluated on the Framingham Heart Study — two datasets with different feature spaces, demographic profiles, and class distributions. The pipeline investigates domain shift, applies mitigation strategies, and quantifies their effect on out-of-distribution (OOD) performance.

**Core finding:** Models achieving ROC-AUC of 0.86–0.90 in-domain degraded to 0.66–0.69 on Framingham. No single mitigation strategy resolved this — each created a different precision/recall tradeoff.

---

## Datasets

| Dataset | Source | Patients | Features | Positive Rate | Role |
|---|---|---|---|---|---|
| UCI Heart Disease Multi-Center | [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) | 920 | 11 clinical | 55.3% | In-distribution (train/val) |
| Framingham Heart Study | [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) | 4,240 | 15 epidemiological | 15.2% | Out-of-distribution (test) |

Loaded via [`kagglehub`](https://github.com/Kaggle/kagglehub). Cross-domain evaluation uses 4 shared features: `age`, `sex`, `cholesterol`, `blood_pressure`.

---

## Project Structure

```
health-risk-prediction-distribution-shift/
│
├── CS4082_Health_Risk_Prediction.ipynb   # Full pipeline — all phases in one notebook
│
├── report/                               # ACL-format technical report (PDF)
├── poster/                               # A1/A0 research poster (PDF)
├── slides/                               # Presentation slides
│
├── index.html                            # Portfolio website (linked at end)
│
├── requirements.txt
└── README.md
```

---
## Notebook Structure

| Phase | Description |
|---|---|
| 1 — Data Preparation | Load datasets, align feature spaces, handle missing values, encode, scale |
| 2 — EDA | Distribution comparison, class imbalance, KS tests, domain shift evidence |
| 3 — Baseline Training (In-Domain) | Train 7 models on UCI full features (80/20 split), default hyperparameters |
| 4 — Baseline OOD Evaluation | Test all models on Framingham using shared features — quantify shift cost |
| 5 — In-Domain Improvement | GridSearchCV tuning, class weighting, SMOTE on UCI |
| 6 — Threshold Tuning | Optimal threshold from UCI val applied to Framingham for OOD robustness |
| 7 — Pooled Training | Train on UCI + partial Framingham to reduce domain gap |
| 8 — Final Comparison | Full metric table across all strategies |
| 9 — Limitations | Honest accounting of experimental constraints |
| 10 — Conclusion | Key findings and practical implications |

---

## Results - Final Comparison (All Strategies)

| Strategy | Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline OOD | AdaBoost | 0.2767 | 0.4674 | 0.3476 | 0.6848 |
| Baseline OOD | SVM (balanced) | — | — | 0.3610 | — |
| Tuned OOD | Logistic Regression | 0.2645 | 0.4969 | 0.3452 | 0.6830 |
| Tuned OOD | Gradient Boosting | 0.2535 | 0.5606 | 0.3491 | 0.6774 |
| **Threshold Tuned** | **Random Forest** | 0.2249 | **0.7438** | 0.3453 | 0.6882 |
| Pooled Training | Logistic Regression | **0.4125** | 0.1710 | 0.2418 | **0.6925** |
| Pooled Training | Gradient Boosting | 0.3623 | 0.1295 | 0.1908 | 0.6911 |
| Pooled Training | Random Forest | 0.3016 | 0.1969 | 0.2382 | 0.6617 |

**Key takeaway by objective:**
- **Maximize recall** (don't miss cases) → Threshold Tuning: Recall = 0.744
- **Maximize precision** (minimize false alarms) → Pooled Training: Precision = 0.413  
- **Balanced F1** → SVM with class weighting: F1 = 0.361
- **ROC-AUC stable** (~0.66–0.69) across all strategies — domain shift degrades calibration more than discrimination

---

## Setup

```bash
git clone https://github.com/[your-username]/health-risk-prediction-distribution-shift.git
cd health-risk-prediction-distribution-shift
pip install -r requirements.txt
```

Open the notebook in Google Colab and run all cells top to bottom.

---

## Team Contributions

| Member | Contributions |
|---|---|
| [Judy Abuquta] | |
| [Nadine ElHaddad] | |
| [Lama Alturki] | |
| [Yara Mohamed] | |

---

## Links

- 📄 [Technical Report (PDF)](report/CS4082_Final_Report.pdf)
- 🌐 [Portfolio Website](https://judyabuquta.github.io/CS4082_health-prediction-distribution-shift/)
- 🎥 [Poster (PDF)](report/poster.pdf)
- 📄 [Slides (PDF)](report/CS4082_FinalPresentation.pdf)
---

## Instructor

**Dr. Naila Marir** — namarir@effatuniversity.edu.sa  
Effat University · Computer Science Department

---

## License

For academic use only. All dataset rights belong to their respective owners.
