"""
Data Registration
-----------------
Registers the tourism dataset that is versioned inside the GitHub repository
(tourism_project/data/tourism.csv):
  1. loads the CSV straight from the repository data folder,
  2. validates the schema against the data dictionary,
  3. validates the target variable,
  4. prints a short summary of the registered data.

The script exits with a non-zero code when validation fails, so the
GitHub Actions pipeline stops before any downstream job runs.
"""
import sys
from pathlib import Path

import pandas as pd

# Paths are resolved relative to this file, so the script behaves the same
# in Google Colab and on the GitHub Actions runner
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

TARGET = "ProdTaken"

# Columns defined in the data dictionary of the project
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    # 1. Load the dataset from the repository data folder
    if not DATA_PATH.exists():
        sys.exit(f"[ERROR] Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded from: {DATA_PATH.relative_to(PROJECT_DIR.parent).as_posix()}")

    # 2. Schema validation: every column of the data dictionary must be present
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_cols:
        sys.exit(f"[ERROR] Missing expected columns: {missing_cols}")
    print(f"Schema check passed: all {len(EXPECTED_COLUMNS)} expected columns are present.")
    if extra_cols:
        print(f"Extra columns found (dropped later in data preparation): {extra_cols}")

    # 3. Target validation: ProdTaken must be binary (0 = No, 1 = Yes)
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        sys.exit(f"[ERROR] Target '{TARGET}' must only contain 0 and 1.")
    print(f"Target check passed: '{TARGET}' is binary (0/1).")

    # 4. Short summary of the registered dataset
    print("\n================ Dataset summary ================")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print(f"Duplicate CustomerIDs: {df['CustomerID'].duplicated().sum()}")
    print(f"Missing values (all columns): {int(df.isna().sum().sum())}")

    print("\nColumn data types:")
    print(df[EXPECTED_COLUMNS].dtypes.value_counts().to_string())

    print(f"\nTarget distribution ({TARGET}):")
    target_dist = pd.concat(
        [df[TARGET].value_counts(), df[TARGET].value_counts(normalize=True).mul(100).round(2)],
        axis=1,
        keys=["count", "percent"],
    )
    print(target_dist.to_string())

    print("\nCategorical columns and their values:")
    for col in df[EXPECTED_COLUMNS].select_dtypes(include="object").columns:
        print(f"  {col}: {df[col].value_counts().to_dict()}")

    print("\nNumeric columns summary:")
    numeric_cols = df[EXPECTED_COLUMNS].drop(columns=["CustomerID", TARGET]).select_dtypes("number")
    print(numeric_cols.describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())

    print("\nDataset registered successfully.")


if __name__ == "__main__":
    main()
