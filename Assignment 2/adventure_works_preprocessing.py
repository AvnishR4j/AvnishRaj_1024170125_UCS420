"""Assignment 2 - Adventure Works customer preprocessing and similarity analysis.

Place the Kaggle CSV file in this folder, or pass its path:
    python3 adventure_works_preprocessing.py path/to/AWCustomers.csv

The script uses all rows from the provided dataset. If no CSV is available, it runs
on a tiny built-in sample only to demonstrate the complete solution flow.
"""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


SELECTED_FEATURES = [
    "MaritalStatus",
    "Gender",
    "YearlyIncome",
    "TotalChildren",
    "NumberChildrenAtHome",
    "Education",
    "Occupation",
    "HomeOwnerFlag",
    "NumberCarsOwned",
    "CommuteDistance",
    "Region",
    "Age",
    "BikeBuyer",
]

ATTRIBUTE_TYPES = {
    "MaritalStatus": "Discrete, Nominal",
    "Gender": "Discrete, Nominal",
    "YearlyIncome": "Continuous, Ratio",
    "TotalChildren": "Discrete, Ratio",
    "NumberChildrenAtHome": "Discrete, Ratio",
    "Education": "Discrete, Ordinal",
    "Occupation": "Discrete, Nominal",
    "HomeOwnerFlag": "Discrete, Nominal/Binary",
    "NumberCarsOwned": "Discrete, Ratio",
    "CommuteDistance": "Discrete, Ordinal",
    "Region": "Discrete, Nominal",
    "Age": "Continuous, Ratio",
    "BikeBuyer": "Discrete, Nominal/Binary target",
}

EDUCATION_ORDER = {
    "Partial High School": 1,
    "High School": 2,
    "Partial College": 3,
    "Bachelors": 4,
    "Graduate Degree": 5,
}


def heading(title):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")


def find_csv_from_args_or_folder():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])

    folder = Path(__file__).parent
    preferred_names = [
        "AWCustomers.csv",
        "AdvWorksCusts.csv",
        "AdventureWorksCustomers.csv",
        "customers.csv",
    ]
    for name in preferred_names:
        candidate = folder / name
        if candidate.exists():
            return candidate

    csv_files = sorted(folder.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return None


def sample_dataset():
    return pd.DataFrame(
        {
            "CustomerID": [11000, 11001, 11002, 11003],
            "Title": [None, None, None, None],
            "FirstName": ["Jon", "Eugene", "Ruben", "Christy"],
            "LastName": ["Yang", "Huang", "Torres", "Zhu"],
            "BirthDate": ["1966-04-08", "1965-05-14", "1965-08-12", "1968-02-15"],
            "MaritalStatus": ["M", "S", "M", "S"],
            "Gender": ["M", "M", "M", "F"],
            "YearlyIncome": [90000, 60000, 60000, np.nan],
            "TotalChildren": [2, 3, 3, 0],
            "NumberChildrenAtHome": [0, 3, 3, 0],
            "Education": ["Bachelors", "Bachelors", "Bachelors", "Partial College"],
            "Occupation": ["Professional", "Professional", "Professional", "Clerical"],
            "HomeOwnerFlag": [1, 0, 1, 0],
            "NumberCarsOwned": [0, 1, 1, 1],
            "CommuteDistance": ["1-2 Miles", "0-1 Miles", "2-5 Miles", "5-10 Miles"],
            "Region": ["Pacific", "Pacific", "Pacific", "Europe"],
            "BikeBuyer": [1, 1, 1, 0],
        }
    )


def load_data():
    csv_path = find_csv_from_args_or_folder()
    if csv_path and csv_path.exists():
        print(f"Using dataset: {csv_path}")
        return pd.read_csv(csv_path)

    print("No Adventure Works CSV found. Running on built-in demo rows.")
    print("For final submission output, place the full Kaggle CSV in this folder and rerun.")
    return sample_dataset()


def add_age_if_needed(df):
    df = df.copy()
    if "Age" not in df.columns and "BirthDate" in df.columns:
        birth_dates = pd.to_datetime(df["BirthDate"], errors="coerce")
        today = pd.Timestamp.today()
        df["Age"] = ((today - birth_dates).dt.days / 365.25).round(1)
    return df


def select_features(df):
    df = add_age_if_needed(df)
    available_features = [column for column in SELECTED_FEATURES if column in df.columns]
    selected = df[available_features].copy()
    return selected


def print_attribute_types(columns):
    for column in columns:
        print(f"{column}: {ATTRIBUTE_TYPES.get(column, 'Type depends on dataset values')}")


def commute_to_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    numbers = [float(number) for number in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan
    if "+" in text:
        return numbers[0]
    if len(numbers) >= 2:
        return sum(numbers[:2]) / 2
    return numbers[0]


def handle_null_values(df):
    cleaned = df.copy()
    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            mode = cleaned[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[column] = cleaned[column].fillna(fill_value)
    return cleaned


def add_discretized_columns(df):
    transformed = df.copy()
    for column in ["YearlyIncome", "Age"]:
        if column in transformed.columns:
            transformed[f"{column}_Bin"] = pd.qcut(
                transformed[column],
                q=min(4, transformed[column].nunique()),
                duplicates="drop",
                labels=False,
            )

    if "CommuteDistance" in transformed.columns:
        transformed["CommuteDistance_Value"] = transformed["CommuteDistance"].map(commute_to_number)
        transformed["CommuteDistance_Bin"] = pd.cut(
            transformed["CommuteDistance_Value"],
            bins=[-0.1, 1, 2, 5, 10, np.inf],
            labels=["0-1", "1-2", "2-5", "5-10", "10+"],
        )
    return transformed


def min_max_normalize(series):
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def standardize(series):
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def transform_input(df):
    cleaned = handle_null_values(df)
    transformed = add_discretized_columns(cleaned)

    if "Education" in transformed.columns:
        transformed["Education_Ordinal"] = transformed["Education"].map(EDUCATION_ORDER).fillna(0)

    numeric_columns = [
        column for column in transformed.columns if pd.api.types.is_numeric_dtype(transformed[column])
    ]
    target_columns = [column for column in ["BikeBuyer"] if column in numeric_columns]
    numeric_features = [column for column in numeric_columns if column not in target_columns]

    numeric_scaled = pd.DataFrame(index=transformed.index)
    for column in numeric_features:
        numeric_scaled[f"{column}_MinMax"] = min_max_normalize(transformed[column].astype(float))
        numeric_scaled[f"{column}_Standard"] = standardize(transformed[column].astype(float))

    categorical_columns = [
        column
        for column in transformed.columns
        if column not in numeric_columns and column not in {"BikeBuyer"}
    ]
    categorical_encoded = pd.get_dummies(
        transformed[categorical_columns].astype(str), prefix=categorical_columns, dtype=int
    )

    final_input = pd.concat([numeric_scaled, categorical_encoded], axis=1)
    if "BikeBuyer" in transformed.columns:
        final_input["BikeBuyer"] = transformed["BikeBuyer"].astype(int)
    return cleaned, transformed, final_input


def simple_matching(row_a, row_b):
    return (row_a == row_b).mean()


def jaccard_similarity(row_a, row_b):
    a = row_a.astype(bool)
    b = row_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return np.logical_and(a, b).sum() / union


def cosine_similarity(row_a, row_b):
    a = row_a.astype(float).to_numpy()
    b = row_b.astype(float).to_numpy()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def main():
    raw_df = load_data()

    heading("Part I(a): Selected attributes")
    selected_df = select_features(raw_df)
    print(selected_df.columns.tolist())

    heading("Part I(b): New DataFrame with selected attributes")
    print(selected_df.head())
    print("Shape:", selected_df.shape)

    heading("Part I(c): Data value type of each selected attribute")
    print_attribute_types(selected_df.columns)

    heading("Part II(a): Handling null values")
    cleaned_df, transformed_df, final_input = transform_input(selected_df)
    print(cleaned_df.head())
    print("Null values after cleaning:")
    print(cleaned_df.isna().sum())

    heading("Part II(b-d): Normalization, discretization and standardization")
    preview_columns = [
        column
        for column in transformed_df.columns
        if column.endswith("_Bin") or column.endswith("_Value")
    ]
    print(transformed_df[preview_columns].head() if preview_columns else "No discretized columns found.")
    print(final_input.filter(regex="MinMax|Standard").head())

    heading("Part II(e): Binarization / One Hot Encoding")
    print(final_input.head())
    print("Transformed input shape:", final_input.shape)

    heading("Part III(a): Similarity between first two transformed objects")
    feature_only_input = final_input.drop(columns=["BikeBuyer"], errors="ignore")
    if len(feature_only_input) >= 2:
        row_1 = feature_only_input.iloc[0]
        row_2 = feature_only_input.iloc[1]
        print("Simple Matching Similarity:", simple_matching(row_1.round(6), row_2.round(6)))
        print("Jaccard Similarity:", jaccard_similarity(row_1, row_2))
        print("Cosine Similarity:", cosine_similarity(row_1, row_2))
    else:
        print("At least two rows are required for similarity calculation.")

    heading("Part III(b): Correlation between Commute Distance and Yearly Income")
    if {"CommuteDistance", "YearlyIncome"}.issubset(cleaned_df.columns):
        corr_df = cleaned_df.copy()
        corr_df["CommuteDistance_Value"] = corr_df["CommuteDistance"].map(commute_to_number)
        print(corr_df[["CommuteDistance", "CommuteDistance_Value", "YearlyIncome"]].head())
        print("Correlation:", corr_df["CommuteDistance_Value"].corr(corr_df["YearlyIncome"]))
    else:
        print("Required columns are not available in this dataset.")


if __name__ == "__main__":
    main()
