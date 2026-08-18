"""Assignment 3 - Pandas solutions.

Run:
    python3 assignment_3_pandas.py
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def heading(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def question_1():
    heading("Q1: Create the dataset")
    data = {
        "Tid": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Refund": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No"],
        "Marital Status": [
            "Single",
            "Married",
            "Single",
            "Married",
            "Divorced",
            "Married",
            "Divorced",
            "Single",
            "Married",
            "Single",
        ],
        "Taxable Income": ["125K", "100K", "70K", "120K", "95K", "60K", "220K", "85K", "75K", "90K"],
        "Cheat": ["No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes"],
    }
    df = pd.DataFrame(data)
    print(df)
    return df


def question_2(df):
    heading("Q2: Locate row 0, 4, 7 and 8")
    print(df.loc[[0, 4, 7, 8]])


def question_3(df):
    heading("Q3: DataFrame navigation")
    print("\nQ3(1): Select rows from index 3 to 7")
    print(df.loc[3:7])

    print("\nQ3(2): Select rows from index 4 to 8, and columns 2 to 4")
    print(df.iloc[4:9, 2:5])

    print("\nQ3(3): Select all rows with column index 1 to 3, including 3")
    print(df.iloc[:, 1:4])


def ensure_iris_csv():
    DATA_DIR.mkdir(exist_ok=True)
    iris_path = DATA_DIR / "iris.csv"
    if not iris_path.exists():
        iris_df = pd.DataFrame(
            {
                "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4],
                "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6, 3.9],
                "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7],
                "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4],
                "species": ["Iris-setosa"] * 6,
            }
        )
        iris_df.to_csv(iris_path, index=False)
    return iris_path


def question_4():
    heading("Q4: Read a CSV file and display first five rows")
    iris_path = ensure_iris_csv()
    iris_df = pd.read_csv(iris_path)
    print("CSV path:", iris_path)
    print(iris_df.head())
    return iris_df


def question_5(iris_df):
    heading("Q5: Delete row 4 and column 3 from the CSV DataFrame")
    result = iris_df.drop(index=4).drop(columns=iris_df.columns[3])
    print(result)


def create_employees_csv():
    DATA_DIR.mkdir(exist_ok=True)
    employees_path = DATA_DIR / "employees.csv"
    employees = pd.DataFrame(
        {
            "Employee_ID": [101, 102, 103, 104, 105],
            "Name": ["Alice", "Bob", "Charlie", "Diana", "Edward"],
            "Department": ["HR", "IT", "IT", "Marketing", "Sales"],
            "Age": [29, 34, 41, 28, 38],
            "Salary": [50000, 70000, 65000, 55000, 60000],
            "Years_of_Experience": [4, 8, 10, 3, 12],
            "Joining_Date": ["2020-03-15", "2017-07-19", "2013-06-01", "2021-02-10", "2010-11-25"],
            "Gender": ["Female", "Male", "Male", "Female", "Male"],
            "Bonus": [5000, 7000, 6000, 4500, 5000],
            "Rating": [4.5, 4.0, 3.8, 4.7, 3.5],
        }
    )
    employees.to_csv(employees_path, index=False)
    return employees_path


def performance_category(rating):
    if rating >= 4.5:
        return "Excellent"
    if rating >= 4.0:
        return "Good"
    return "Average"


def question_6():
    heading("Q6: Employee dataset operations")
    employees_path = create_employees_csv()
    df = pd.read_csv(employees_path)

    print("\nQ6(a): Shape")
    print(df.shape)

    print("\nQ6(b): Summary with data types and non-null counts")
    df.info()

    print("\nQ6(c): Descriptive statistics")
    print(df.describe(include="all"))

    print("\nQ6(d): First 5 rows and last 3 rows")
    print("First 5 rows:")
    print(df.head())
    print("Last 3 rows:")
    print(df.tail(3))

    print("\nQ6(e): Statistics")
    print("Average salary:", df["Salary"].mean())
    print("Total bonus paid:", df["Bonus"].sum())
    print("Youngest employee age:", df["Age"].min())
    print("Highest performance rating:", df["Rating"].max())

    print("\nQ6(f): Sort by Salary in descending order")
    print(df.sort_values(by="Salary", ascending=False))

    print("\nQ6(g): Add Performance_Category column")
    df["Performance_Category"] = df["Rating"].apply(performance_category)
    print(df[["Name", "Rating", "Performance_Category"]])

    print("\nQ6(h): Identify missing values")
    print(df.isnull().sum())

    print("\nQ6(i): Rename Employee_ID column to ID")
    df = df.rename(columns={"Employee_ID": "ID"})
    print(df.head())

    print("\nQ6(j): Employees with more than 5 years of experience")
    print(df[df["Years_of_Experience"] > 5])

    print("\nQ6(j): Employees belonging to the IT department")
    print(df[df["Department"] == "IT"])

    print("\nQ6(k): Add Tax column with 10 percent of Salary deducted")
    df["Tax"] = df["Salary"] * 0.10
    print(df[["Name", "Salary", "Tax"]])

    print("\nQ6(l): Save modified DataFrame to a new CSV file")
    output_path = DATA_DIR / "employees_modified.csv"
    df.to_csv(output_path, index=False)
    print("Saved:", output_path)


def main():
    df = question_1()
    question_2(df)
    question_3(df)
    iris_df = question_4()
    question_5(iris_df)
    question_6()


if __name__ == "__main__":
    main()
