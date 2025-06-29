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

def preprocess_data(df_path='data/student_data.csv', target_col_name='passed', test_size=0.25, random_state=42):
    """
    Loads, preprocesses the student data, and splits it into train/test sets.
    """
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)

    # Attempt to load data, handle if not found for now
    try:
        df = pd.read_csv(df_path)
        print(f"Successfully loaded {df_path}")
    except FileNotFoundError:
        print(f"Warning: {df_path} not found. Creating a dummy DataFrame for structure demonstration.")
        # Create a dummy DataFrame that resembles the expected structure
        data = {
            'age': np.random.randint(15, 20, size=100),
            'Medu': np.random.randint(0, 5, size=100), # Mother's education
            'Fedu': np.random.randint(0, 5, size=100), # Father's education
            'studytime': np.random.randint(1, 5, size=100), # 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, 4 - >10 hours
            'failures': np.random.randint(0, 4, size=100), # number of past class failures
            'absences': np.random.randint(0, 93, size=100),
            'G1': np.random.randint(0, 20, size=100),
            'G2': np.random.randint(0, 20, size=100),
            'G3': np.random.randint(0, 101, size=100), # Final grade (0-100 to allow for threshold 50)
            'sex': np.random.choice(['F', 'M'], size=100),
            'address': np.random.choice(['U', 'R'], size=100), # Urban/Rural
            'famsize': np.random.choice(['LE3', 'GT3'], size=100), # Family size
            'Pstatus': np.random.choice(['T', 'A'], size=100), # Parent's cohabitation status
            'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], size=100),
            'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], size=100),
            'reason': np.random.choice(['home', 'reputation', 'course', 'other'], size=100),
            'guardian': np.random.choice(['mother', 'father', 'other'], size=100),
            'schoolsup': np.random.choice(['yes', 'no'], size=100), # Extra educational support
            'famsup': np.random.choice(['yes', 'no'], size=100), # Family educational support
            'paid': np.random.choice(['yes', 'no'], size=100), # Extra paid classes
            'activities': np.random.choice(['yes', 'no'], size=100), # Extra-curricular activities
            'nursery': np.random.choice(['yes', 'no'], size=100), # Attended nursery school
            'higher': np.random.choice(['yes', 'no'], size=100), # Wants to take higher education
            'internet': np.random.choice(['yes', 'no'], size=100), # Internet access at home
            'romantic': np.random.choice(['yes', 'no'], size=100) # With a romantic relationship
        }
        # For simplicity, let's assume 'G3' is always present in the dummy data for target creation.
        # If 'passed' is the target, it will be handled by create_target_variable.
        # If student_data.csv has a 'passed' column, it will be used.
        # Otherwise, 'G3' will be used to create 'passed'.
        df = pd.DataFrame(data)
        # Introduce some missing values for testing imputation
        for col in ['age', 'G1', 'G2', 'G3', 'Mjob', 'Fjob']: # mix of numeric and categorical
            if col in df.columns:
                idx = df.sample(frac=0.05, random_state=random_state).index # 5% missing
                df.loc[idx, col] = np.nan
        print("Created a dummy DataFrame with potential numeric and categorical columns, including G3 for target derivation if 'passed' is absent.")

    print("\nFirst 5 rows of the dataframe:")
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


    # Identify numeric and categorical columns
    # G3 should not be in potential_numeric_cols as it's used for target or already dropped
    potential_numeric_cols = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    # Ensure G3 is not accidentally included if it was not dropped properly or if it's not the grade_col_name
    if 'G3' in X.columns:
        print("Warning: G3 column found in features X after target creation. Dropping it.")
        X = X.drop(columns=['G3'], errors='ignore')

    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) and col in potential_numeric_cols]
    # For any other columns that are numeric but not in our 'potential' list, let's add them if they are truly numeric
    # This makes it more robust if new numeric columns are added to the dataset
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and col not in numeric_cols and col not in potential_numeric_cols:
            # Add to numeric_cols only if it's not an ID or something obviously non-feature like
            # For now, we assume all other numeric columns are features. This might need refinement.
            print(f"Automatically identified additional numeric column: {col}")
            numeric_cols.append(col)


    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    print(f"\nIdentified Numeric columns: {numeric_cols}")
    print(f"Identified Categorical columns: {categorical_cols}")

    # Encode categorical features
    # Example ordinal mapping (if any, needs to be defined based on data specifics)
    # For now, assume all categorical are nominal and use one-hot encoding
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print("\nApplied one-hot encoding to categorical features.")
    else:
        print("\nNo categorical features to encode.")

    # Scale numeric features
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        print("\nScaled numeric features using StandardScaler.")
    else:
        print("\nNo numeric features to scale.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"\nData split into train and test sets. Test size: {test_size}, Random state: {random_state}")

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Save the scaler, numeric_cols, categorical_cols, and X_train column names
    # These will be used in evaluate.py to ensure consistent preprocessing
    os.makedirs('models', exist_ok=True) # Ensure models directory exists
    if numeric_cols: # Only save scaler if it was used
        joblib.dump(scaler, 'models/scaler.pkl')
        print("Saved StandardScaler to models/scaler.pkl")
    joblib.dump(numeric_cols, 'models/numeric_cols.pkl')
    print("Saved numeric_cols to models/numeric_cols.pkl")
    joblib.dump(categorical_cols, 'models/categorical_cols.pkl')
    print("Saved categorical_cols to models/categorical_cols.pkl")
    joblib.dump(X_train.columns.tolist(), 'models/x_train_columns.pkl') # Save column names after dummification
    print("Saved X_train_columns to models/x_train_columns.pkl")


    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    print("Running data_processing.py as main script...")
    # This will use the dummy data generation if 'data/student_data.csv' is not found.
    # To test with actual data, ensure 'data/student_data.csv' exists.

    # Create a dummy student_data.csv for the test to run, if it doesn't exist
    # This ensures the script can run end-to-end for structural validation
    # even if the actual data file isn't provided in the `data/` directory yet.
    dummy_data_path = 'data/student_data.csv'
    if not os.path.exists(dummy_data_path):
        print(f"Creating a dummy '{dummy_data_path}' for test run as it's missing.")
        os.makedirs('data', exist_ok=True) # Ensure data directory exists
        dummy_df_data = {
            'age': [18, 17, 15, 15, 16, 19, 20, 17, 18, 15] * 10, # 100 students
            'Medu': [4, 1, 1, 4, 3, 2, 1, 4, 3, 2] * 10,
            'Fedu': [4, 2, 1, 2, 3, 1, 1, 3, 2, 4] * 10,
            'studytime': [2, 2, 2, 3, 1, 1, 2, 3, 2, 4] * 10,
            'failures': [0, 0, 3, 0, 0, 1, 0, 0, 0, 0] * 10,
            'absences': [6, 4, 10, 2, 4, 0, 2, 6, 0, 0] * 10,
            'G1': [5, 5, 7, 15, 6, 10, 12, 14, 10, 15] * 10, # Scores typically 0-20
            'G2': [6, 5, 8, 14, 10, 9, 12, 14, 10, 15] * 10, # Scores typically 0-20
            'G3': [np.random.randint(0, 101) for _ in range(100)], # Scores 0-100 for 'passed' threshold 50
            'sex': ['F', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'M'] * 10,
            'address': ['U', 'U', 'U', 'R', 'U', 'R', 'U', 'U', 'R', 'U'] * 10,
            'famsize': ['GT3', 'GT3', 'LE3', 'GT3', 'GT3', 'LE3', 'GT3', 'LE3', 'GT3', 'LE3'] * 10,
            'Pstatus': ['A', 'T', 'T', 'T', 'A', 'T', 'A', 'T', 'A', 'T'] * 10,
            'Mjob': ['at_home', 'at_home', 'at_home', 'health', 'services', 'other', 'teacher', 'health', 'services', 'other'] * 10,
            'Fjob': ['teacher', 'other', 'other', 'services', 'other', 'teacher', 'services', 'other', 'teacher', 'other'] * 10,
            'reason': ['course', 'course', 'other', 'home', 'reputation', 'course', 'home', 'reputation', 'home', 'course'] * 10,
            'guardian': ['mother', 'father', 'mother', 'father', 'mother', 'father', 'mother', 'father', 'mother', 'father'] * 10,
            'schoolsup': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no'] * 10,
            'famsup': ['no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no'] * 10,
            'paid': ['no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'] * 10,
            'activities': ['no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'] * 10,
            'nursery': ['yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'no'] * 10,
            'higher': ['yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes'] * 10,
            'internet': ['no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes'] * 10,
            'romantic': ['no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'] * 10
            # 'passed': [0, 0, 1, 1, 1, 0, 1, 1, 1, 1] * 10 # Example 'passed' column
        }
        dummy_df = pd.DataFrame(dummy_df_data)
        # Introduce some NaN values into the dummy CSV to test imputation logic
        for col in ['G1', 'Mjob']: # one numeric, one categorical
            idx = dummy_df.sample(frac=0.1, random_state=1).index # 10% missing to test imputation
            dummy_df.loc[idx, col] = np.nan
        dummy_df.to_csv(dummy_data_path, index=False)
        print(f"Dummy '{dummy_data_path}' created with some missing values for testing.")

    X_train_main, X_test_main, y_train_main, y_test_main = preprocess_data(df_path=dummy_data_path)

    print("\n--- Main Script Test Output ---")
    print(f"X_train shape from main: {X_train_main.shape}")
    print(f"X_test shape from main: {X_test_main.shape}")
    print(f"y_train counts from main:\n{y_train_main.value_counts(normalize=True)}")
    print(f"y_test counts from main:\n{y_test_main.value_counts(normalize=True)}")
    print("Data processing script run complete.")
