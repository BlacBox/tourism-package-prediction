# Visit with Us - Wellness Tourism Package Prediction (MLOps)

An end-to-end MLOps pipeline that predicts whether a customer will purchase the
new **Wellness Tourism Package** before the sales team contacts them.

- **Live app (Streamlit Community Cloud):** https://blacbox-tourism-package-prediction.streamlit.app
- **CI/CD:** GitHub Actions (`.github/workflows/pipeline.yml`) runs on every push to `main`

## Pipeline

```
push to main
   |
   v
[register-dataset] --> [data-prep] --> [model-training] --> commit model to main --> Streamlit app redeploys
 schema checks         cleaning +       GridSearchCV +
 + data summary        train/test       MLflow tracking
                       split (artifact)
```

| Job | Script | Output |
|---|---|---|
| `register-dataset` | `tourism_project/model_building/data_register.py` | validated dataset + summary (artifact `registered-data`) |
| `data-prep` | `tourism_project/model_building/prep.py` | `Xtrain/Xtest/ytrain/ytest.csv` (artifact `data-splits`) |
| `model-training` | `tourism_project/model_building/train.py` | tuned XGBoost model committed to `tourism_project/deployment/`, MLflow runs (artifact `mlflow-tracking`) |

## Repository structure

```
.
├── .github/workflows/pipeline.yml          # CI/CD pipeline (GitHub Actions)
├── README.md
└── tourism_project/
    ├── data/
    │   └── tourism.csv                     # registered dataset (versioned in the repo)
    ├── model_building/
    │   ├── data_register.py                # schema validation + data summary
    │   ├── prep.py                         # cleaning + stratified train/test split
    │   └── train.py                        # tuning, MLflow tracking, evaluation, model saving
    ├── deployment/
    │   ├── app.py                          # Streamlit app
    │   ├── requirements.txt                # app dependencies (Streamlit Community Cloud)
    │   ├── best_tourism_model_v1.joblib    # committed by the pipeline
    │   └── model_metrics.json              # committed by the pipeline
    └── requirements.txt                    # pipeline dependencies
```

## Model

- Preprocessing: `StandardScaler` for numeric features, `OneHotEncoder` for categorical features
- Algorithm: `XGBClassifier` with `scale_pos_weight` for the class imbalance (~19% buyers)
- Tuning: `GridSearchCV`, 48 parameter combinations, stratified 5-fold CV, selected on F1-score
- Tracking: every candidate is logged to MLflow as a nested run (parameters + CV scores)

## Retraining

Push a new version of `tourism_project/data/tourism.csv` (or any code change) to `main`,
or click **Run workflow** in the Actions tab. The pipeline retrains the model, commits it
back to `main`, and Streamlit Community Cloud redeploys the app with the new model.
