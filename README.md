# Tumor Spheroid Time-Series ML & Visualization

A compact toolkit to **predict relapse vs. control** of tumor spheroids from radiomics/time-series features (e.g., days 0–14) and to produce publication-ready figures used in the manuscript.

It contains:
[Figure 1.pdf](https://github.com/user-attachments/files/22549137/Figure.1.pdf)



- **`MLTimeSeriesModel.py`** – a configurable ML pipeline with variance filtering, scaling, **SMOTE**, multiple **feature-selection** strategies, and a zoo of classifiers; includes cross-validation with **95% CIs**, bootstrapped evaluation, probability-based ROC utilities, and global seed helpers for reproducibility.
- **`Visualization.py`** – a plotting suite for confusion matrices, ROC curves (with optimal threshold), probability distributions, **accuracy vs. Day-of-Relapse (DoR)** with weighted exponential fits, and stacked **SCP** (Share of Controlled Proportion) bar charts for treatment arms (dose, temperature, time).
- **`Manuscript_codes_fixed2.ipynb`** – an analysis notebook (used in the manuscript).

---

## ✨ Features

- **End-to-end pipeline**  
  VarianceThreshold → StandardScaler → SMOTE → Feature Selection → Model.
- **Flexible feature selection**  
  ANOVA, mutual information, PCA, Lasso, tree-based importance, RFE, sequential.
- **Wide model zoo**  
  Logistic Regression, SVM, Random Forest, XGBoost, LDA, NB, KNN, Gradient/Ada/HistGB, Ridge, MLP.
- **Cross-validation with 95% CIs**  
  Reports mean ± CI for AUC, Accuracy, F1 (Student-t).
- **Bootstrap testing**  
  Optional resampling to get robust CIs for test metrics.
- **Manuscript-grade plots**  
  Confusion matrices, ROC with Youden-J threshold, accuracy vs DoR with log-linear fits, SCP stacked bars.

---

## 📦 Installation

```bash
# Python 3.10+ recommended
pip install -U numpy pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn statsmodels
# Optional (for deterministic seeding)
pip install torch
