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

## Results

### Phase 3 — In-Domain Baseline (UCI Validation Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| AdaBoost | 0.8370 | 0.8273 | 0.8922 | **0.8585** | **0.9036** |
| Gradient Boosting | 0.8207 | 0.8165 | 0.8725 | 0.8436 | 0.8910 |
| Random Forest | 0.8043 | 0.8190 | 0.8333 | 0.8309 | 0.8847 |
| Logistic Regression | ~0.800 | ~0.810 | ~0.840 | ~0.820 | ~0.880 |
| SVM | ~0.795 | ~0.800 | ~0.845 | ~0.820 | ~0.875 |
| KNN | ~0.790 | ~0.800 | ~0.835 | ~0.815 | ~0.865 |
| Naive Bayes | ~0.788 | 0.7944 | 0.8235 | 0.8134 | 0.8648 |

### Phase 4 — Baseline OOD Evaluation (Framingham)

> In-domain AUC of 0.86–0.90 dropped to 0.66–0.68. F1 collapsed from ~0.82–0.86 to 0.32–0.35.

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| **AdaBoost** | 0.2767 | 0.4674 | **0.3476** | **0.6848** |
| Gradient Boosting | 0.2629 | 0.4969 | 0.3439 | 0.6833 |
| Logistic Regression | 0.2592 | 0.4720 | 0.3346 | 0.6775 |
| Naive Bayes | 0.2408 | 0.5186 | 0.3289 | 0.6730 |
| SVM | 0.2467 | 0.5575 | 0.3421 | 0.6695 |
| KNN | 0.2403 | 0.5373 | 0.3321 | 0.6669 |
| Random Forest | 0.2396 | 0.4907 | 0.3220 | 0.6570 |

### Phase 8 — Final Comparison (All Strategies)

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
- 🌐 [Portfolio Website](#)
- 🎥 [Poster (PDF)](report/)
- 📄 [Slides (PDF)](report/)
---

## Instructor

**Dr. Naila Marir** — namarir@effatuniversity.edu.sa  
Effat University · Computer Science Department

---

## License

For academic use only. All dataset rights belong to their respective owners.
