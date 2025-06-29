import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import numpy as np # Make sure numpy is imported for get_feature_importances if used there
from sklearn.model_selection import train_test_split # For splitting data
# No longer directly using preprocess_data from here for the main X_test, y_test
# from src.data_processing import preprocess_data, create_target_variable # Keep create_target_variable if used for y_test derivation from raw

# We need to load the raw data and apply transformations
# Assuming data_processing.py's create_target_variable and missing value handling logic
# should be applied to raw data before splitting and transforming test set.
# For simplicity, we'll assume the data loading part of preprocess_data can be reused or simplified.
# The core idea is to get raw X_test and y_test, then apply saved transformers.

try:
    from src.data_processing import create_target_variable, preprocess_data # preprocess_data needed for main guard
except ModuleNotFoundError:
    from data_processing import create_target_variable, preprocess_data # preprocess_data needed for main guard


def load_model(path):
    """Loads a pickled model from disk."""
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return None
    print(f"Loading model from {path}...")
    return joblib.load(path)

def get_feature_importances(model, feature_names, top_n=10):
    """Extracts and returns feature importances or coefficients."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        top_indices = sorted_indices[:top_n]
        df = pd.DataFrame({'feature': [feature_names[i] for i in top_indices], 'importance': importances[top_indices]})
        return df.sort_values(by='importance', ascending=False)
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        coefficients = np.abs(model.coef_[0]) # Assuming binary classification, take first set of coefs
        sorted_indices = np.argsort(coefficients)[::-1]
        top_indices = sorted_indices[:top_n]
        # Ensure feature_names matches the number of coefficients
        if len(coefficients) != len(feature_names):
            print(f"Warning: Number of coefficients ({len(coefficients)}) does not match number of feature names ({len(feature_names)}). Skipping LR coefficients.")
            return pd.DataFrame() # Return empty if mismatch
        df = pd.DataFrame({'feature': [feature_names[i] for i in top_indices], 'abs_coefficient': coefficients[top_indices]})
        return df.sort_values(by='abs_coefficient', ascending=False)
    return pd.DataFrame() # Return empty DataFrame if no importances/coefficients

def evaluate_models():
    """
    Loads trained models, evaluates them on the test set, and saves metrics.
    """
    print("Starting model evaluation process...")

    # Create directories if they don't exist
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('models', exist_ok=True) # Ensure models dir exists, though train.py should create it
    os.makedirs('data', exist_ok=True)

    # 1. Load raw data and prepare test set
    print("Loading raw data for evaluation...")
    data_path = 'data/student_data.csv' # Define data_path
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Creating a dummy DataFrame for evaluation structure.")
        # Recreate a similar dummy DataFrame structure as in data_processing.py
        # This is for the script to run, actual evaluation needs real data and consistent train/test split.
        dummy_data = {
            'age': np.random.randint(15, 20, size=30), 'Medu': np.random.randint(0, 5, size=30),
            'Fedu': np.random.randint(0, 5, size=30), 'studytime': np.random.randint(1, 5, size=30),
            'failures': np.random.randint(0, 4, size=30), 'absences': np.random.randint(0, 93, size=30),
            'G1': np.random.randint(0, 20, size=30), 'G2': np.random.randint(0, 20, size=30),
            'G3': np.random.randint(0, 101, size=30), # Crucial for target (0-100 to allow for threshold 50)
            'sex': np.random.choice(['F', 'M'], size=30), 'address': np.random.choice(['U', 'R'], size=30),
            'famsize': np.random.choice(['LE3', 'GT3'], size=30), 'Pstatus': np.random.choice(['T', 'A'], size=30),
            'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], size=30),
            'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], size=30),
            'reason': np.random.choice(['home', 'reputation', 'course', 'other'], size=30),
            'guardian': np.random.choice(['mother', 'father', 'other'], size=30),
            'schoolsup': np.random.choice(['yes', 'no'], size=30), 'famsup': np.random.choice(['yes', 'no'], size=30),
            'paid': np.random.choice(['yes', 'no'], size=30), 'activities': np.random.choice(['yes', 'no'], size=30),
            'nursery': np.random.choice(['yes', 'no'], size=30), 'higher': np.random.choice(['yes', 'no'], size=30),
            'internet': np.random.choice(['yes', 'no'], size=30),'romantic': np.random.choice(['yes', 'no'], size=30)
        }
        df = pd.DataFrame(dummy_data)
        # Minimal missing value handling for the dummy test data
        for col in ['G1', 'Mjob', 'G3']: # G3 is important for target
             if col in df.columns:
                idx = df.sample(frac=0.05, random_state=42).index
                df.loc[idx, col] = np.nan
                if pd.api.types.is_numeric_dtype(df[col]): df[col].fillna(df[col].median(), inplace=True)
                else: df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {data_path}")
        # Minimal missing value handling for real data - should mirror train.py's data_processing
        # For simplicity, applying median/mode imputation similar to data_processing.py's fallback
        # A more robust solution would save the imputer objects from training or use a pipeline
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Ideally, use median from training data if saved, else from current data
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    # Ideally, use mode from training data if saved, else from current data
                    df[column].fillna(df[column].mode()[0], inplace=True)
        print("Applied basic missing value imputation to loaded data.")


    # Create target variable 'passed' using the G3 threshold of 50
    df = create_target_variable(df, target_col_name='passed', grade_col_name='G3', threshold=50)
    if 'passed' not in df.columns:
        raise ValueError("Target column 'passed' could not be created or found in evaluation data.")

    y = df['passed']
    X_raw = df.drop(columns=['passed', 'G3'], errors='ignore') # Drop G3 as it's not a feature

    # Load preprocessing artifacts
    print("Loading preprocessing artifacts...")
    try:
        scaler = joblib.load('models/scaler.pkl')
        numeric_cols = joblib.load('models/numeric_cols.pkl')
        categorical_cols = joblib.load('models/categorical_cols.pkl')
        x_train_columns = joblib.load('models/x_train_columns.pkl')
        print("Successfully loaded scaler, numeric_cols, categorical_cols, and x_train_columns.")
    except FileNotFoundError:
        print("Error: Preprocessing artifact(s) not found. Make sure train.py has been run successfully.")
        print("Skipping model evaluation as critical preprocessing info is missing.")
        return pd.DataFrame(), {} # Return empty results

    # IMPORTANT: We need to split the raw data (X_raw, y) to get X_test and y_test
    # This split must use the same random_state and test_size as in data_processing.py during training
    # to ensure we're evaluating on the correct unseen portion.
    # This assumes that data_processing.py was run with test_size=0.25, random_state=42 for the main split.
    # A more robust way would be to save X_test, y_test from train.py, but this is a common approach.
    _, X_test_raw, _, y_test = train_test_split(X_raw, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Raw X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")


    # Apply transformations to X_test_raw
    # 1. One-hot encode categorical features
    if categorical_cols:
        X_test_encoded = pd.get_dummies(X_test_raw, columns=categorical_cols, drop_first=True)
    else:
        X_test_encoded = X_test_raw.copy()
    print("Applied one-hot encoding to test data's categorical features.")

    # 2. Align columns with X_train_columns (handles missing/extra columns after dummification)
    X_test_aligned = X_test_encoded.reindex(columns=x_train_columns, fill_value=0)
    print("Aligned test data columns with training data columns.")

    # 3. Scale numeric features using the loaded scaler
    if numeric_cols and scaler: # Check if scaler was loaded (i.e. numeric_cols was not empty during training)
        # Ensure only numeric_cols that are present in X_test_aligned are scaled
        cols_to_scale = [col for col in numeric_cols if col in X_test_aligned.columns]
        if cols_to_scale:
            X_test_aligned[cols_to_scale] = scaler.transform(X_test_aligned[cols_to_scale])
            print("Scaled numeric features in test data using loaded scaler.")
        else:
            print("No numeric columns to scale in test data or numeric_cols list is empty.")
    else:
        print("No numeric features to scale (either numeric_cols is empty or scaler was not loaded).")

    X_test = X_test_aligned # This is the fully preprocessed test set
    print(f"Final X_test shape for evaluation: {X_test.shape}")


    # 2. Define model paths and names
    model_paths = {
        "Logistic Regression": "models/best_logistic_regression.pkl",
        "Decision Tree": "models/best_decision_tree.pkl",
        "Random Forest": "models/best_random_forest.pkl"
    }

    all_metrics = []
    feature_importances_dfs = {}

    # 3. Load models and evaluate
    for model_name, model_path in model_paths.items():
        print(f"\nEvaluating {model_name}...")
        model = load_model(model_path)

        if model is None:
            print(f"Skipping {model_name} due to missing model file.")
            # Add dummy metrics to prevent crash if a model is missing
            metrics = {
                'model': model_name, 'accuracy': np.nan, 'precision': np.nan,
                'recall': np.nan, 'f1_score': np.nan, 'roc_auc': np.nan
            }
            all_metrics.append(metrics)
            if model_name in ["Decision Tree", "Random Forest"]:
                 feature_importances_dfs[f"{model_name.lower().replace(' ', '_')}_feature_importances"] = pd.DataFrame()
            elif model_name == "Logistic Regression":
                 feature_importances_dfs["lr_coefficients"] = pd.DataFrame()
            continue


        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Metrics for {model_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        metrics = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        all_metrics.append(metrics)

        # Feature importances/coefficients
        # X_test.columns contains the feature names after preprocessing (e.g. one-hot encoding)
        importances_df = get_feature_importances(model, X_test.columns.tolist(), top_n=10)
        if not importances_df.empty:
            print(f"\nTop 10 features/coefficients for {model_name}:")
            print(importances_df)
            if model_name in ["Decision Tree", "Random Forest"]:
                filename = f"reports/figures/{model_name.lower().replace(' ', '_')}_feature_importances.csv"
                feature_importances_dfs[f"{model_name.lower().replace(' ', '_')}_feature_importances"] = importances_df
            elif model_name == "Logistic Regression":
                filename = f"reports/figures/lr_coefficients.csv"
                feature_importances_dfs["lr_coefficients"] = importances_df

            importances_df.to_csv(filename, index=False)
            print(f"Saved feature importances/coefficients to {filename}")
        else:
            print(f"No feature importances or coefficients available for {model_name}.")


    # 4. Save all metrics to a CSV file
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = "reports/figures/metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nAll model metrics saved to {metrics_csv_path}")

    print("\nModel evaluation complete.")
    return metrics_df, feature_importances_dfs

if __name__ == '__main__':
    import numpy as np # for get_feature_importances
    print("Running evaluate.py as main script...")

    # To run evaluate.py independently for testing, we need dummy model files
    # if train.py hasn't been run.
    # The preprocess_data function will create dummy data if 'data/student_data.csv' is missing.
    # We also need to ensure dummy model files exist if actual ones are not there.

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True) # For preprocess_data called by evaluate_models

    # Create dummy model files if real ones aren't there from train.py
    # This requires scikit-learn and models.py to be importable
    # preprocess_data (called below and in evaluate_models) will handle dummy student_data.csv creation
    dummy_data_path = 'data/student_data.csv' # Define for preprocess_data call
    try:
        from src.models import get_logistic_regression, get_decision_tree, get_random_forest
    except ModuleNotFoundError:
        from models import get_logistic_regression, get_decision_tree, get_random_forest

    # Call preprocess_data once to get a consistent X_test_dummy structure for fitting dummy models.
    # This will also create the dummy 'data/student_data.csv' using data_processing.py's logic if it doesn't exist.
    # The evaluate_models() function will then use this same CSV.
    print("Preparing dummy data structures for potential dummy model creation in evaluate.py's main guard...")
    _, X_test_dummy, _, y_test_dummy = preprocess_data(df_path=dummy_data_path, test_size=0.25, random_state=42) # Ensure params match evaluate_models
    print(f"Dummy X_test_dummy shape for fitting dummy models: {X_test_dummy.shape}")

    dummy_models_to_create = {
        "models/best_logistic_regression.pkl": get_logistic_regression(random_state=42).fit(X_test_dummy, y_test_dummy),
        "models/best_decision_tree.pkl": get_decision_tree(random_state=42).fit(X_test_dummy, y_test_dummy),
        "models/best_random_forest.pkl": get_random_forest(random_state=42, n_estimators=10).fit(X_test_dummy, y_test_dummy)
    }

    for path, model_instance in dummy_models_to_create.items():
        if not os.path.exists(path):
            print(f"Creating dummy model file: {path} for test run.")
            joblib.dump(model_instance, path)

    metrics_summary, importances_summary = evaluate_models()

    print("\n--- Main Script Test Output ---")
    if not metrics_summary.empty:
        print("Metrics Summary:")
        print(metrics_summary)
    else:
        print("No metrics were generated.")

    if importances_summary:
        print("\nFeature Importances/Coefficients Summary:")
        for name, df in importances_summary.items():
            if not df.empty:
                print(f"--- {name} ---")
                print(df)
            else:
                print(f"--- {name} (no data) ---")

    print("evaluate.py script run complete.")
