 # Predicting Treatment Response of Tumor Spheroids from AI-Driven Image Analysis of Post-Treatment Dynamics
 contact: pejman.shojaee@tu-dresden.de
 pejman.shojaee@htw-dresden.de
 
 Affiliation: TU Dresden
 
 HTW Dresden – University of Applied Sciences

## Background

The project implements an AI-driven image analysis pipeline for predicting long-term relapse of tumor spheroids from early post-treatment brightfield image data.


It contains:

- **`MLTimeSeriesModel.py`** – a configurable ML pipeline with variance filtering, scaling, **SMOTE**, multiple **feature-selection** strategies, and a zoo of classifiers; includes cross-validation with **95% CIs**, bootstrapped evaluation, probability-based ROC utilities, and global seed helpers for reproducibility.
- **`Visualization.py`** – a plotting suite for confusion matrices, ROC curves (with optimal threshold), probability distributions, **accuracy vs. Day-of-Relapse (DoR)** with weighted exponential fits, and stacked **SCP** (Share of Controlled Proportion) bar charts for treatment arms (dose, temperature, time).
- **`Manuscript_codes_fixed2.ipynb`** – an analysis notebook (used in the manuscript).


Here, we present an end-to-end machine learning framework that predicts whether individual spheroids relapse or remain controlled, based solely on early post-treatment imaging features.
Our workflow integrates AI image analysis, feature extraction, feature selection, and classification models to infer long-term treatment outcomes.

## 📦 Installation

```bash
# Python 3.10+ recommended
pip install -U numpy pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn statsmodels
# Optional (for deterministic seeding)
pip install torch
```

Notes:
- All CSVs in data/ are exactly those used in the manuscript models (training and test).
- Splits used in the paper are under data/splits/.

## Quick start
You can easily use the uploaded time range data and use it to visualize the results. You can also use your own data and the pipeline to calculate the classified target, such as relapsed and controlled cases.

```bash
from MLTimeSeriesModel import MLTimeSeriesModel
from Visualization import Visualization
import pandas as pd

# Load your dataset
df = pd.read_csv("data/short_final_with_updated_diagnosis.csv")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Initialize pipeline
model = MLTimeSeriesModel()
model.select_feature_selection("SelectKBest_f_classif")
model.select_model("RandomForest")
model.build_pipeline()

# Cross-validation
results = model.cross_validate(X, y)
print(results["mean_auc"], results["auc_confidence_interval"])

# Visualization example
viz = Visualization(output_dir="results", y_true=y, y_pred=None)
viz.plot_confusion_matrix()
```

## maintainers
- Pejman Shojaee ([@pejmanshojam31](https://github.com/pejmanshojam31)) — lead
- Tom Bischopink ([@tomb556](https://github.com/tombk556))
