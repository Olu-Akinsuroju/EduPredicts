import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_target_variable(df, target_col_name='passed', grade_col_name='G3', threshold=50):
    """
    Creates the target variable 'passed'.
    If target_col_name exists, it's used. Otherwise, it's derived from grade_col_name.
    """
    # Clean column names
    df.columns = df.columns.str.strip()
    print("Available columns for target creation:", df.columns.tolist())

    if target_col_name not in df.columns:
        if grade_col_name in df.columns:
            print(f"'{target_col_name}' not found; creating from '{grade_col_name}' >= {threshold}.")
            df[target_col_name] = (df[grade_col_name] >= threshold).astype(int)
        else:
            raise ValueError(f"Neither '{target_col_name}' nor '{grade_col_name}' found in DataFrame.")
    else:
        print(f"Using existing '{target_col_name}' column.")
    return df


def add_interaction_features(df):
    """
    Adds custom interaction features based on educational insights.
    """
    # 1. Study Time × Motivation Levels
    df['study_high']   = df['Study_Time_per_Week'] * df['Motivation_Level_High']
    df['study_med']    = df['Study_Time_per_Week'] * df['Motivation_Level_Medium']
    df['study_low']    = df['Study_Time_per_Week'] * df['Motivation_Level_Low']

    # 2. Previous Grade × Motivation Levels
    df['grade_high']   = df['Previous_Grade'] * df['Motivation_Level_High']
    df['grade_med']    = df['Previous_Grade'] * df['Motivation_Level_Medium']
    df['grade_low']    = df['Previous_Grade'] * df['Motivation_Level_Low']

    # 3. Absences × Internet Access
    df['abs_int_net']  = df['Absences'] * df['Internet_Access_Yes']

    # 4. Parent Education × Study Time
    df['study_parent_pri'] = df['Study_Time_per_Week'] * df['Parent_Education_Level_Primary']
    df['study_parent_sec'] = df['Study_Time_per_Week'] * df['Parent_Education_Level_Secondary']
    df['study_parent_ter'] = df['Study_Time_per_Week'] * df['Parent_Education_Level_Tertiary']

    # 5. Previous Grade × Age
    df['grade_age']    = df['Previous_Grade'] * df['Age']

    return df


def preprocess_data(df_path='data/student_data.csv', target_col_name='passed', test_size=0.25, random_state=42):
    """
    Loads, cleans, derives target, adds interaction features, and splits student data into train/test sets.
    """
    os.makedirs(os.path.dirname(df_path), exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(df_path)
        print(f"Loaded data from '{df_path}'.")
    except Exception as e:
        print(f"Error loading '{df_path}': {e}")
        return None, None, None, None

    # Optional: preview data
    print("\nData preview:")
    print(df.head())

    # Derive target variable
    df = create_target_variable(df, target_col_name=target_col_name, grade_col_name='G3', threshold=50)

    # Add interaction features
    df = add_interaction_features(df)
    print("Added interaction features:", [col for col in df.columns if col in [
        'study_high','study_med','study_low',
        'grade_high','grade_med','grade_low',
        'abs_int_net',
        'study_parent_pri','study_parent_sec','study_parent_ter',
        'grade_age']])

    # Prepare features and target
    y = df[target_col_name]
    X = df.drop(columns=[target_col_name, 'G3'], errors='ignore')
    print(f"Features shape after dropping target & G3: {X.shape}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Split data: X_train {X_train.shape}, X_test {X_test.shape}")
    print("Target distribution (train):\n", y_train.value_counts(normalize=True))
    print("Target distribution (test):\n", y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("Running data preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Preprocessing complete.")
