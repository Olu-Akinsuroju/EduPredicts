import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def create_target_variable(df, target_col_name='passed', grade_col_name='G3', threshold=50):
    """
    Creates the target variable 'passed'.
    If target_col_name exists, it's used. Otherwise, it's derived from grade_col_name.
    """
    if target_col_name not in df.columns:
        if grade_col_name in df.columns:
            print(f"'{target_col_name}' column not found. Creating it based on '{grade_col_name}' >= {threshold}.")
            df[target_col_name] = (df[grade_col_name] >= threshold).astype(int)
        else:
            raise ValueError(f"Neither '{target_col_name}' nor '{grade_col_name}' found in DataFrame. Cannot create target variable.")
    else:
        print(f"Using existing '{target_col_name}' column as target variable.")
    return df

# Placeholder for user's feature interaction logic
def add_interaction_features(df):
    """
    Adds interaction and engineered features to the DataFrame.
    This is a placeholder and should be replaced with actual feature engineering logic.
    """
    print("Applying placeholder interaction features...")
    df_out = df.copy()

    # Ensure base columns exist, if not, create dummy versions or skip feature
    # Base features expected: Study_Time_per_Week, Previous_Grade, Absences, Internet_Access, Parent_Education_Level, Age

    # study_high, study_med, study_low (based on Study_Time_per_Week)
    if 'Study_Time_per_Week' in df_out.columns:
        # Placeholder: quantiles
        low_quantile = df_out['Study_Time_per_Week'].quantile(0.33)
        high_quantile = df_out['Study_Time_per_Week'].quantile(0.66)
        df_out['study_low'] = (df_out['Study_Time_per_Week'] < low_quantile).astype(int)
        df_out['study_med'] = ((df_out['Study_Time_per_Week'] >= low_quantile) & (df_out['Study_Time_per_Week'] < high_quantile)).astype(int)
        df_out['study_high'] = (df_out['Study_Time_per_Week'] >= high_quantile).astype(int)
    else:
        print("Warning: 'Study_Time_per_Week' not found for study interactions. Skipping.")
        df_out['study_low'] = 0
        df_out['study_med'] = 0
        df_out['study_high'] = 0


    # grade_high, grade_med, grade_low (based on Previous_Grade)
    if 'Previous_Grade' in df_out.columns:
        # Placeholder: quantiles
        low_g_quantile = df_out['Previous_Grade'].quantile(0.33)
        high_g_quantile = df_out['Previous_Grade'].quantile(0.66)
        df_out['grade_low'] = (df_out['Previous_Grade'] < low_g_quantile).astype(int)
        df_out['grade_med'] = ((df_out['Previous_Grade'] >= low_g_quantile) & (df_out['Previous_Grade'] < high_g_quantile)).astype(int)
        df_out['grade_high'] = (df_out['Previous_Grade'] >= high_g_quantile).astype(int)
    else:
        print("Warning: 'Previous_Grade' not found for grade interactions. Skipping.")
        df_out['grade_low'] = 0
        df_out['grade_med'] = 0
        df_out['grade_high'] = 0

    # abs_int_net (Absences * Internet_Access - assuming Internet_Access is 'Yes'/'No')
    if 'Absences' in df_out.columns and 'Internet_Access' in df_out.columns:
        internet_numeric = df_out['Internet_Access'].replace({'Yes': 1, 'No': 0}).fillna(0) # Handle potential NaNs if not imputed yet
        df_out['abs_int_net'] = df_out['Absences'] * internet_numeric
    else:
        print("Warning: 'Absences' or 'Internet_Access' not found for abs_int_net. Skipping.")
        df_out['abs_int_net'] = 0

    # study_parent_pri, study_parent_sec, study_parent_ter (Study_Time_per_Week * Parent_Education_Level category)
    # Placeholder: assumes Parent_Education_Level has 'Primary', 'Secondary', 'Tertiary' like values after potential OHE or mapping
    # This is complex to do robustly as a placeholder without knowing exact PEL values.
    # For simplicity, let's create dummy interactions with Study_Time_per_Week if PEL exists.
    if 'Study_Time_per_Week' in df_out.columns and 'Parent_Education_Level' in df_out.columns:
        # Example: Interaction with a specific category if it exists (e.g., 'College')
        df_out['study_parent_pri'] = df_out['Study_Time_per_Week'] * (df_out['Parent_Education_Level'] == 'HighSchool').astype(int) # Placeholder
        df_out['study_parent_sec'] = df_out['Study_Time_per_Week'] * (df_out['Parent_Education_Level'] == 'College').astype(int)    # Placeholder
        df_out['study_parent_ter'] = df_out['Study_Time_per_Week'] * (df_out['Parent_Education_Level'] == 'Masters').astype(int)  # Placeholder
    else:
        print("Warning: 'Study_Time_per_Week' or 'Parent_Education_Level' not found for study_parent interactions. Skipping.")
        df_out['study_parent_pri'] = 0
        df_out['study_parent_sec'] = 0
        df_out['study_parent_ter'] = 0

    # grade_age (Previous_Grade / Age)
    if 'Previous_Grade' in df_out.columns and 'Age' in df_out.columns:
        # Adding a small epsilon to age to avoid division by zero if age can be 0 (unlikely for this dataset)
        df_out['grade_age'] = df_out['Previous_Grade'] / (df_out['Age'] + 1e-6)
    else:
        print("Warning: 'Previous_Grade' or 'Age' not found for grade_age. Skipping.")
        df_out['grade_age'] = 0

    print(f"Columns after adding interactions: {df_out.columns.tolist()}")
    return df_out


def preprocess_data(df_path='data/student_data.csv', target_col_name='passed', test_size=0.25, random_state=42):
    """
    Loads, performs minimal preprocessing, and splits data.
    Feature engineering (interactions, scaling, encoding) is now handled in the training pipeline.
    """
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)

    # Attempt to load data, handle if not found for now
    try:
        # Specify tab separation for the actual data file
        df = pd.read_csv(df_path, sep='\t')
        print(f"Successfully loaded {df_path} (using tab separator).")
    except FileNotFoundError:
        print(f"Warning: {df_path} not found. Creating a dummy DataFrame for structure demonstration.")
        # Create a dummy DataFrame that aligns with new feature engineering requirements
        data = {
            # User-specified numeric features
            'Age': np.random.randint(15, 20, size=100),
            'Absences': np.random.randint(0, 93, size=100),
            'Study_Time_per_Week': np.random.uniform(1, 10, size=100).round(1), # e.g., hours
            'Previous_Grade': np.random.randint(0, 101, size=100), # Assuming 0-100 scale

            # User-specified binary-encode features
            'Gender': np.random.choice(['Male', 'Female'], size=100),
            'Extra_Courses': np.random.choice(['Yes', 'No'], size=100),
            'Internet_Access': np.random.choice(['Yes', 'No'], size=100),

            # User-specified one-hot encode features
            'Motivation_Level': np.random.choice(['Low', 'Medium', 'High', 'Very High'], size=100),
            'Parent_Education_Level': np.random.choice(['None', 'HighSchool', 'College', 'Masters', 'PhD'], size=100),

            # Target derivation column (will be dropped from features)
            'G3': np.random.randint(0, 101, size=100),

            # Other potential features from original dummy data (can be pruned if not needed)
            'Medu': np.random.randint(0, 5, size=100), # Mother's education (can be mapped to Parent_Education_Level or kept)
            'Fedu': np.random.randint(0, 5, size=100), # Father's education
            'failures': np.random.randint(0, 4, size=100),
            'schoolsup': np.random.choice(['yes', 'no'], size=100),
            'famsup': np.random.choice(['yes', 'no'], size=100),
            'activities': np.random.choice(['yes', 'no'], size=100),
            'higher': np.random.choice(['yes', 'no'], size=100), # Wants to take higher education
            'romantic': np.random.choice(['yes', 'no'], size=100)
        }
        df = pd.DataFrame(data)
        # Introduce some missing values for testing imputation
        # For new named columns and some old ones if kept
        cols_for_nan = ['Age', 'Previous_Grade', 'Gender', 'Motivation_Level', 'Parent_Education_Level', 'Medu', 'failures']
        for col in cols_for_nan:
            if col in df.columns: # Ensure column exists before trying to make NaNs
                idx = df.sample(frac=0.05, random_state=random_state).index # 5% missing
                df.loc[idx, col] = np.nan
        print("Created a new dummy DataFrame aligned with feature engineering specs, including G3 for target derivation.")

    print("\nFirst 5 rows of the (potentially dummy) dataframe:")
    print(df.head())
    print("\nData types of columns:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe(include='all'))

    print("\nMissing values count before handling:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # Handle missing values
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].isnull().sum() / len(df) < 0.01: # if <1% missing, drop rows
                print(f"Dropping rows with missing values in column '{column}' (less than 1% missing).")
                df.dropna(subset=[column], inplace=True)
            else: # otherwise impute
                if pd.api.types.is_numeric_dtype(df[column]):
                    median_val = df[column].median()
                    print(f"Imputing missing values in numeric column '{column}' with median: {median_val}.")
                    df[column].fillna(median_val, inplace=True)
                else:
                    mode_val = df[column].mode()[0]
                    print(f"Imputing missing values in categorical column '{column}' with mode: {mode_val}.")
                    df[column].fillna(mode_val, inplace=True)

    print("\nMissing values count after handling:")
    missing_values_after = df.isnull().sum()
    print(missing_values_after[missing_values_after > 0])
    if missing_values_after.sum() == 0:
        print("All missing values handled.")

    # Create target variable 'passed'
    # User specified threshold is 50 for G3
    df = create_target_variable(df, target_col_name=target_col_name, grade_col_name='G3', threshold=50)

    if target_col_name not in df.columns:
         raise ValueError(f"Target column '{target_col_name}' could not be created or found.")

    y = df[target_col_name]
    # Features X should not contain the target variable or the grade variable G3 used to create it.
    X = df.drop(columns=[target_col_name, 'G3'], errors='ignore')
    print(f"Shape of X after dropping target and G3: {X.shape}")

    # Explicit encoding, scaling, and artifact saving are removed.
    # This will be handled by ColumnTransformer in train.py.

    # Split data into raw train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"\nData split into raw train and test sets. Test size: {test_size}, Random state: {random_state}")

    print(f"X_train raw shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test raw shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    print("Running data_processing.py as main script...")
    # This will use the dummy data generation if 'data/student_data.csv' is not found.
    # To test with actual data, ensure 'data/student_data.csv' exists.

    dummy_data_path = 'data/student_data.csv'
    if not os.path.exists(dummy_data_path):
        print(f"Creating a dummy '{dummy_data_path}' for test run as it's missing.")
        os.makedirs('data', exist_ok=True) # Ensure data directory exists
        # Updated dummy data to reflect new feature engineering names
        dummy_df_data = {
            'Age': np.random.randint(15, 20, size=100),
            'Absences': np.random.randint(0, 93, size=100),
            'Study_Time_per_Week': np.random.uniform(1, 10, size=100).round(1),
            'Previous_Grade': np.random.randint(0, 101, size=100),
            'Gender': np.random.choice(['Male', 'Female'], size=100),
            'Extra_Courses': np.random.choice(['Yes', 'No'], size=100),
            'Internet_Access': np.random.choice(['Yes', 'No'], size=100),
            'Motivation_Level': np.random.choice(['Low', 'Medium', 'High', 'Very High'], size=100),
            'Parent_Education_Level': np.random.choice(['None', 'HighSchool', 'College', 'Masters', 'PhD'], size=100),
            'G3': [np.random.randint(0, 101) for _ in range(100)], # Target derivation
            # Keeping a few old ones for broader initial dataset, can be pruned by ColumnTransformer
            'Medu': np.random.randint(0, 5, size=100),
            'Fedu': np.random.randint(0, 5, size=100),
            'failures': np.random.randint(0, 4, size=100),
        }
        dummy_df = pd.DataFrame(dummy_df_data)
        # Introduce some NaN values
        cols_for_nan_main = ['Age', 'Previous_Grade', 'Gender', 'Motivation_Level', 'Medu']
        for col in cols_for_nan_main:
            idx = dummy_df.sample(frac=0.1, random_state=1).index
            dummy_df.loc[idx, col] = np.nan
        dummy_df.to_csv(dummy_data_path, index=False)
        print(f"Dummy '{dummy_data_path}' created with new feature names and some missing values for testing.")

    X_train_main, X_test_main, y_train_main, y_test_main = preprocess_data(df_path=dummy_data_path)

    print("\n--- Main Script Test Output (Raw Data) ---")
    print(f"X_train shape from main: {X_train_main.shape}")
    print(f"X_test shape from main: {X_test_main.shape}")
    print(f"y_train counts from main:\n{y_train_main.value_counts(normalize=True)}")
    print(f"y_test counts from main:\n{y_test_main.value_counts(normalize=True)}")
    print("Data processing script run complete.")
