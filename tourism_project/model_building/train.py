"""
Model Training with Experiment Tracking
---------------------------------------
  1. loads the train/test splits produced by the data preparation job,
  2. defines a preprocessing + XGBoost pipeline and a hyperparameter grid,
  3. tunes the model with GridSearchCV (5-fold stratified cross-validation),
  4. logs every tuned parameter combination and its scores to MLflow,
  5. evaluates the best model on the train and test sets,
  6. saves the best model to tourism_project/deployment/ so the workflow can
     commit it to the repository for the Streamlit app.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path.cwd()  # the data-splits artifact is downloaded here
MODEL_DIR = PROJECT_DIR / "deployment"
MODEL_PATH = MODEL_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "tourism-package-prediction"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Hyperparameter grid: 2 x 3 x 2 x 2 x 2 = 48 candidate models
PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
}

# F1 drives model selection (balances finding buyers and not wasting calls);
# the other scores are tracked for every candidate as well
SCORING = ["f1", "recall", "precision", "roc_auc"]
REFIT_METRIC = "f1"

# Keep the console output readable (no "View run ... at" line for each of the 48 runs)
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")


def evaluate(model, X, y):
    """Return the main classification metrics of a fitted model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }


def main():
    # 1. Load the splits produced by the data preparation job
    Xtrain = pd.read_csv(DATA_DIR / "Xtrain.csv")
    Xtest = pd.read_csv(DATA_DIR / "Xtest.csv")
    ytrain = pd.read_csv(DATA_DIR / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(DATA_DIR / "ytest.csv").squeeze("columns")
    print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")

    # 2. Preprocessing: scale numeric features, one-hot encode categorical ones
    #    (unknown categories at prediction time are ignored instead of failing)
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )

    # Class imbalance (~19% buyers): weight the positive class by neg/pos ratio
    scale_pos_weight = round((ytrain == 0).sum() / (ytrain == 1).sum(), 3)
    classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    pipeline = make_pipeline(preprocessor, classifier)

    # 3. Hyperparameter tuning with stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=cv,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI} | experiment: {EXPERIMENT_NAME}")

    with mlflow.start_run(run_name="xgboost-gridsearch") as parent_run:
        grid.fit(Xtrain, ytrain)
        results = grid.cv_results_
        n_candidates = len(results["params"])

        # 4a. Log every tuned parameter combination as a nested run
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"candidate-{i + 1:02d}", nested=True):
                mlflow.log_params({k.replace("xgbclassifier__", ""): v for k, v in params.items()})
                cv_scores = {f"cv_mean_{m}": float(results[f"mean_test_{m}"][i]) for m in SCORING}
                cv_scores[f"cv_std_{REFIT_METRIC}"] = float(results[f"std_test_{REFIT_METRIC}"][i])
                mlflow.log_metrics(cv_scores)
        print(f"Logged {n_candidates} tuned parameter combinations to MLflow (nested runs).")

        # 4b. Log the search setup and the best parameters on the parent run
        best_params = {k.replace("xgbclassifier__", ""): v for k, v in grid.best_params_.items()}
        mlflow.log_params({
            "model": "XGBClassifier",
            "cv_folds": cv.get_n_splits(),
            "selection_metric": REFIT_METRIC,
            "n_candidates": n_candidates,
            "scale_pos_weight": scale_pos_weight,
            "param_grid": json.dumps({k.replace("xgbclassifier__", ""): v for k, v in PARAM_GRID.items()}),
        })
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # 5. Evaluate the best model on train and test data
        best_model = grid.best_estimator_
        train_metrics = evaluate(best_model, Xtrain, ytrain)
        test_metrics = evaluate(best_model, Xtest, ytest)
        mlflow.log_metric(f"best_cv_{REFIT_METRIC}", float(grid.best_score_))
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        print("\n================ Best model ================")
        print(f"Best parameters: {best_params}")
        print(f"Best CV {REFIT_METRIC}: {grid.best_score_:.4f}")

        comparison = pd.DataFrame({"train": train_metrics, "test": test_metrics}).round(4)
        print("\nTrain vs test performance:")
        print(comparison.to_string())

        print("\nClassification report (train):")
        print(classification_report(ytrain, best_model.predict(Xtrain), digits=4))
        print("Classification report (test):")
        print(classification_report(ytest, best_model.predict(Xtest), digits=4))
        tn, fp, fn, tp = confusion_matrix(ytest, best_model.predict(Xtest)).ravel()
        print(f"Test confusion matrix -> TN: {tn}  FP: {fp}  FN: {fn}  TP: {tp}")

        # 6. Save the best model where the workflow commits it from
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        metrics_summary = {
            "model": "XGBClassifier",
            "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "best_params": best_params,
            "best_cv_f1": round(float(grid.best_score_), 4),
            "train_metrics": {k: round(v, 4) for k, v in train_metrics.items()},
            "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
            "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgb.__version__},
            "mlflow_run_id": parent_run.info.run_id,
        }
        METRICS_PATH.write_text(json.dumps(metrics_summary, indent=2))
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="model")

        print(f"\nBest model saved to: {MODEL_PATH.relative_to(PROJECT_DIR.parent).as_posix()}")
        print(f"Metrics saved to   : {METRICS_PATH.relative_to(PROJECT_DIR.parent).as_posix()}")
        print(f"MLflow parent run id: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
