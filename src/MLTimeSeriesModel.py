import warnings, os, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (SelectKBest, f_classif, VarianceThreshold, mutual_info_classif, f_classif,
                                       VarianceThreshold, mutual_info_classif, SelectFromModel, RFE,
                                       SequentialFeatureSelector)
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,                                                                     
    roc_auc_score, 
    f1_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve
)
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.utils import resample
import torch
import torch.nn as nn

def set_global_seed(seed: int = 42, deterministic_torch: bool = True):
    # Python hashing & RNGs
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Make threaded BLAS deterministic (keeps tiny numeric drift in check)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Torch (optional)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


class MLTimeSeriesModel:
    def __init__(self, feature_selection_methods=None, models=None, cv_folds=5, variance_threshold=0.01, seed: int = 42):
        """
        Initialize the MLTimeSeriesModel class.
        
        :param feature_selection_methods: Dictionary of feature selection options (name: method).
        :param models: Dictionary of ML models to choose from (name: model).
        :param cv_folds: Number of cross-validation folds (default is 5).
        :param variance_threshold: Threshold for removing low-variance features (default is 0.01).
        """
        self.seed = seed
        set_global_seed(seed)
        self.feature_selection_methods = feature_selection_methods or {
            "SelectKBest_f_classif": SelectKBest(score_func=f_classif, k=50),
            "SelectKBest_mutual_info": SelectKBest(score_func=mutual_info_classif, k=50),
            "PCA": PCA(n_components=50, svd_solver='full'),
            "LassoSelector": SelectFromModel(Lasso(alpha=0.01, random_state=self.seed)),
            "TreeBasedSelector": SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=self.seed)),
            "RecursiveFeatureElimination": RFE(estimator=LogisticRegression(max_iter=5000, random_state=self.seed), n_features_to_select=50),
            "SequentialForwardSelector": SequentialFeatureSelector(estimator=LogisticRegression(max_iter=5000, random_state=self.seed), 
                                                                n_features_to_select=50, direction='forward'),
            "SequentialBackwardSelector": SequentialFeatureSelector(estimator=LogisticRegression(max_iter=5000, random_state=self.seed), 
                                                                    n_features_to_select=50, direction='backward')
        }   
        self.models = models or {
            "LogisticRegression": LogisticRegression(class_weight='balanced', random_state=self.seed, max_iter=5000),
            "SVM": SVC(kernel='rbf', probability=True, random_state=self.seed),
            "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=self.seed),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=self.seed),
            "LDA": LinearDiscriminantAnalysis(),
            "NaiveBayes": GaussianNB(),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "GradientBoosting": GradientBoostingClassifier(random_state=self.seed),
            "AdaBoost": AdaBoostClassifier(random_state=self.seed),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=self.seed),
            "RidgeClassifier": RidgeClassifier(class_weight='balanced', random_state=self.seed),
            "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100,), random_state=self.seed, max_iter=1500),
            "Lasso": Lasso(random_state=self.seed)
        }

        self.cv_folds = cv_folds
        self.variance_threshold = variance_threshold  # Set the variance threshold
        self.selected_feature_method = None
        self.selected_model = None
        self.feature_selector = None
        self.model = None
        self.pipeline = None
        self.tuned_params = None

    def select_feature_selection(self, method_name):
        """
        Select the feature selection method.
        
        :param method_name: Name of the feature selection method to use.
        """
        if method_name in self.feature_selection_methods:
            self.feature_selector = self.feature_selection_methods[method_name]
            self.selected_feature_method = method_name
        else:
            raise ValueError(f"Feature selection method '{method_name}' is not available.")

    def select_model(self, model_name):
        """
        Select the machine learning model.
        
        :param model_name: Name of the model to use.
        """
        if model_name in self.models:
            self.model = self.models[model_name]
            self.selected_model = model_name
        else:
            raise ValueError(f"Model '{model_name}' is not available.")

    def build_pipeline(self):
        """
        Build the preprocessing and modeling pipeline with SMOTE and variance filtering.
        """
        if not self.feature_selector:
            raise ValueError("Feature selection method is not selected.")
        if not self.model:
            raise ValueError("Model is not selected.")
        
        self.pipeline = ImbPipeline([
            ('low_variance_filter', VarianceThreshold(threshold=self.variance_threshold)),  # Remove low-variance features
            ('scaler', StandardScaler()),                # Scale the data
            ('smote', SMOTE(random_state=42)),           # Handle class imbalance
            ('feature_selection', self.feature_selector),  # Feature selection
            ('model', self.model)                        # Model
        ])
    def cross_validate(self, X, y):
        """
        Perform cross-validation and return detailed per-fold information along with
        mean ± 95% confidence intervals for AUC-ROC, Accuracy, and F1 scores.

        Parameters:
            X: Features (DataFrame or ndarray).
            y: Labels (Series or ndarray).

        Returns:
            A dictionary with:
            - 'mean_auc': Mean AUC-ROC score.
            - 'std_auc': Standard deviation of AUC-ROC scores.
            - 'auc_confidence_interval': 95% confidence interval for AUC-ROC.
            - 'mean_accuracy': Mean accuracy.
            - 'std_accuracy': Standard deviation of accuracy.
            - 'accuracy_confidence_interval': 95% confidence interval for accuracy.
            - 'mean_f1': Mean F1 score.
            - 'std_f1': Standard deviation of F1 score.
            - 'f1_confidence_interval': 95% confidence interval for F1.
            - 'fold_details_df': A pandas DataFrame containing per-fold details:
                    * Fold: Fold number.
                    * Train_Count: Number of training samples.
                    * Test_Count: Number of validation samples.
                    * AUC: AUC-ROC score.
                    * Accuracy: Accuracy score.
                    * F1: F1 score.
        """
        if not self.pipeline:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        fold_details = []
        auc_scores = []
        accuracy_scores = []
        f1_scores = []
        
        # Helper for proper indexing whether using pandas or numpy arrays.
        get_subset = lambda data, indices: data.iloc[indices] if hasattr(data, 'iloc') else data[indices]

        for fold, (train_index, valid_index) in enumerate(skf.split(X, y), start=1):
            X_train = get_subset(X, train_index)
            X_valid = get_subset(X, valid_index)
            y_train = get_subset(y, train_index)
            y_valid = get_subset(y, valid_index)
            
            # Fit the pipeline on the training fold.
            self.pipeline.fit(X_train, y_train)
            
            # Get probabilities for AUC; fallback to decision_function if needed.
            if hasattr(self.pipeline.named_steps['model'], "predict_proba"):
                y_proba = self.pipeline.predict_proba(X_valid)[:, 1]
            else:
                y_proba = self.pipeline.decision_function(X_valid)
            
            fold_auc = roc_auc_score(y_valid, y_proba)
            y_pred = self.pipeline.predict(X_valid)
            fold_accuracy = accuracy_score(y_valid, y_pred)
            fold_f1 = f1_score(y_valid, y_pred, average='weighted')
            
            auc_scores.append(fold_auc)
            accuracy_scores.append(fold_accuracy)
            f1_scores.append(fold_f1)
            
            fold_details.append({
                'Fold': fold,
                'Train_Count': len(train_index),
                'Test_Count': len(valid_index),
                'AUC': fold_auc,
                'Accuracy': fold_accuracy,
                'F1': fold_f1
            })
        
        # Function to compute mean, std, and 95% confidence interval for a list of scores.
            def compute_stats(scores_list):
                mean_val = np.mean(scores_list)
                std_val = np.std(scores_list, ddof=1)
                n = len(scores_list)
                t_critical = t.ppf((1 + 0.95) / 2, n - 1)
                margin_error = t_critical * sem(scores_list)
                ci = (mean_val - margin_error, mean_val + margin_error)
                return mean_val, std_val, ci

        mean_auc, std_auc, auc_ci = compute_stats(auc_scores)
        mean_acc, std_acc, acc_ci = compute_stats(accuracy_scores)
        mean_f1, std_f1, f1_ci = compute_stats(f1_scores)
        
        fold_details_df = pd.DataFrame(fold_details)
        
        return {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'auc_confidence_interval': auc_ci,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'accuracy_confidence_interval': acc_ci,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'f1_confidence_interval': f1_ci,
            'fold_details_df': fold_details_df
        }

    def fit(self, X_train, y_train):
        """
        Fit the pipeline on the training data.
        
        :param X_train: Training features (DataFrame or ndarray).
        :param y_train: Training labels (Series or ndarray).
        """
        if not self.pipeline:
            raise ValueError("Pipeline is not built. Call build_pipeline() after selecting feature selection and model.")

        self.pipeline.fit(X_train, y_train)
    def bootstrap_ci(self, metric_values, confidence=0.95):
        """
        Compute the bootstrap confidence interval.
        
        :param metric_values: List or array of metric values (e.g., accuracy scores from bootstrapping).
        :param confidence: Confidence level (default: 95%).
        :return: (lower bound, upper bound)
        """
        n_iterations = 1000  # Number of bootstrap samples
        n_size = len(metric_values)
        
        bootstrap_samples = [np.mean(resample(metric_values, replace=True, n_samples=n_size)) for _ in range(n_iterations)]
        
        lower = np.percentile(bootstrap_samples, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_samples, (1 + confidence) / 2 * 100)
    
        return lower, upper
    def evaluate_bootstrap(self, X_test, y_test):
        """
        Evaluate the pipeline on the test data and compute 95% CI for metrics.
        
        :param X_test: Test features (DataFrame or ndarray).
        :param y_test: Test labels (Series or ndarray).
        :return: Dictionary of evaluation metrics with 95% CI.
        """
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1] if hasattr(self.pipeline.named_steps['model'], "predict_proba") else None

        acc_scores = []
        auc_scores = []

        # Bootstrapping loop
        for _ in range(1000):  # 1000 resamples
            X_resample, y_resample = resample(X_test, y_test, replace=True)
            y_pred_resample = self.pipeline.predict(X_resample)
            y_proba_resample = self.pipeline.predict_proba(X_resample)[:, 1] if y_proba is not None else None
            
            acc_scores.append(accuracy_score(y_resample, y_pred_resample))
            if y_proba_resample is not None:
                auc_scores.append(roc_auc_score(y_resample, y_proba_resample))

        acc_mean = np.mean(acc_scores)
        acc_ci = self.bootstrap_ci(acc_scores)

        auc_mean = np.mean(auc_scores) if y_proba is not None else None
        auc_ci = self.bootstrap_ci(auc_scores) if y_proba is not None else None

        metrics = {
            "accuracy": acc_mean,
            "accuracy_CI": acc_ci,
            "roc_auc": auc_mean,
            "roc_auc_CI": auc_ci,
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=1),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

        return metrics
    
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the pipeline on the test data without bootstrapping.
        
        :param X_test: Test features (DataFrame or ndarray).
        :param y_test: Test labels (Series or ndarray).
        :return: Dictionary of evaluation metrics (single estimate).
        """
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Probability predictions (if supported)
        if hasattr(self.pipeline.named_steps['model'], "predict_proba"):
            y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = None
        
        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        # Compile metrics
        metrics = {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=1),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def predict(self, X):
        """
        Predict using the trained pipeline.
        
        :param X: Features for prediction (DataFrame or ndarray).
        :return: Predicted labels and probabilities (if available).
        """
        y_pred = self.pipeline.predict(X)
        y_proba = self.pipeline.predict_proba(X)[:, 1] if hasattr(self.pipeline.named_steps['model'], "predict_proba") else None
        return y_pred, y_proba
    
    def tune_hyperparameters(self, X, y, param_grid, search_type='grid', n_iter=10, scoring='roc_auc', n_jobs=-1):
        if not self.selected_model:
            raise ValueError("Select a model first using select_model().")

        base_model = self.models[self.selected_model]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])

        if search_type == 'grid':
            search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, cv=self.cv_folds, n_jobs=n_jobs)
        elif search_type == 'random':
            search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=n_iter,
                                        scoring=scoring, cv=self.cv_folds, n_jobs=n_jobs, random_state=42)
        else:
            raise ValueError("search_type must be 'grid' or 'random'")

        search.fit(X, y)
        self.model = search.best_estimator_.named_steps['model']
        self.tuned_params = search.best_params_
        print(f"Best parameters for {self.selected_model}: {self.tuned_params}")