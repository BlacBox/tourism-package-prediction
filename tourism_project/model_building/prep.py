"""
Data Preparation
----------------
  1. loads the dataset directly from the repository data folder,
  2. cleans it (unnecessary columns, label typos, duplicates, missing values),
  3. splits it into stratified training and testing sets,
  4. saves the splits locally as Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv.

The splits are written to the current working directory (the repository root
on the GitHub Actions runner), which is where the workflow picks them up and
passes them to the training job as an artifact.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
OUTPUT_DIR = Path.cwd()

TARGET = "ProdTaken"
ID_COLUMN = "CustomerID"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    # 1. Load the dataset from the repository data folder
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Remove unnecessary columns: the leftover CSV index and the customer ID
    #    (identifiers carry no information about purchase behaviour)
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + [ID_COLUMN]
    df = df.drop(columns=drop_cols)
    print(f"Dropped unnecessary columns: {drop_cols}")

    # 3. Fix inconsistent category labels ('Fe Male' is a typo of 'Female')
    n_typos = int((df["Gender"] == "Fe Male").sum())
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    print(f"Gender: replaced {n_typos} 'Fe Male' values with 'Female'")

    # 4. Remove exact duplicate records (same profile and interaction recorded twice)
    n_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {n_duplicates} duplicate rows")

    # 5. Missing values: none in the current file, but new data may contain
    #    some, so numeric columns get the median and categorical ones the mode
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        for col in df.columns.drop(TARGET):
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    print(f"Missing values imputed: {n_missing}")
    print(f"Clean dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 6. Stratified split keeps the same purchase rate in train and test
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 7. Save the splits locally (uploaded by the workflow as an artifact)
    Xtrain.to_csv(OUTPUT_DIR / "Xtrain.csv", index=False)
    Xtest.to_csv(OUTPUT_DIR / "Xtest.csv", index=False)
    ytrain.to_csv(OUTPUT_DIR / "ytrain.csv", index=False)
    ytest.to_csv(OUTPUT_DIR / "ytest.csv", index=False)

    print("\n================ Train / test split ================")
    print(f"Xtrain: {Xtrain.shape} | ytrain purchase rate: {ytrain.mean():.2%}")
    print(f"Xtest : {Xtest.shape} | ytest purchase rate : {ytest.mean():.2%}")
    print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
