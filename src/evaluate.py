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
    # preprocess_data is needed for the __main__ guard's dummy X_test_dummy generation.
    # create_target_variable is used in evaluate_models.
    # add_interaction_features is now also needed.
    from src.data_processing import create_target_variable, preprocess_data, add_interaction_features
    from src.models import get_logistic_regression, get_decision_tree, get_random_forest # For main guard dummy models
except ModuleNotFoundError:
    from data_processing import create_target_variable, preprocess_data, add_interaction_features
    from models import get_logistic_regression, get_decision_tree, get_random_forest # For main guard dummy models


def load_pipeline(path):
    """Loads a pickled pipeline from disk."""
    if not os.path.exists(path):
        print(f"Pipeline file not found: {path}")
        return None
    print(f"Loading pipeline from {path}...")
    return joblib.load(path)

def get_feature_importances_from_pipeline(pipeline, top_n=10):
    """
    Extracts and returns feature importances or coefficients from a scikit-learn pipeline.
    Assumes the pipeline has a 'preprocessor' (ColumnTransformer) step and a 'classifier' step.
    """
    try:
        preprocessor = pipeline.named_steps.get('preprocessor')
        classifier = pipeline.named_steps.get('classifier')

        if preprocessor is None or classifier is None:
            print("Pipeline does not have 'preprocessor' or 'classifier' steps.")
            return pd.DataFrame()

        # Get feature names after transformation
        try:
            # Ensure all transformers within ColumnTransformer are fitted if they have get_feature_names_out
            # This relies on the preprocessor being a ColumnTransformer
            feature_names_out = preprocessor.get_feature_names_out()
        except Exception as e:
            print(f"Could not get feature names from preprocessor: {e}")
            # Fallback if get_feature_names_out fails or preprocessor is not as expected
            # This part might need adjustment based on actual ColumnTransformer structure
            # For now, we'll return empty if names can't be reliably fetched.
            return pd.DataFrame()

        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            if len(importances) != len(feature_names_out):
                print(f"Mismatch between number of importances ({len(importances)}) and feature names ({len(feature_names_out)}).")
                return pd.DataFrame()
            sorted_indices = np.argsort(importances)[::-1]
            top_indices = sorted_indices[:top_n]
            df = pd.DataFrame({
                'feature': [feature_names_out[i] for i in top_indices],
                'importance': importances[top_indices]
            })
            return df.sort_values(by='importance', ascending=False)
        elif hasattr(classifier, 'coef_'):
            coefficients = np.abs(classifier.coef_[0]) # Assuming binary classification
            if len(coefficients) != len(feature_names_out):
                print(f"Mismatch between number of coefficients ({len(coefficients)}) and feature names ({len(feature_names_out)}).")
                return pd.DataFrame()
            sorted_indices = np.argsort(coefficients)[::-1]
            top_indices = sorted_indices[:top_n]
            df = pd.DataFrame({
                'feature': [feature_names_out[i] for i in top_indices],
                'abs_coefficient': coefficients[top_indices]
            })
            return df.sort_values(by='abs_coefficient', ascending=False)
    except Exception as e:
        print(f"Error extracting feature importances from pipeline: {e}")
    return pd.DataFrame()


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
    data_path = 'data/student_data.csv'
    # Define TEST_SIZE and RANDOM_STATE to be consistent with data_processing.py
    TEST_SIZE = 0.25
    RANDOM_STATE = 42

    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Creating a dummy DataFrame for evaluation structure.")
        # This dummy data structure should now match the one in data_processing.py
        dummy_data = {
            'Age': np.random.randint(15, 20, size=30),
            'Absences': np.random.randint(0, 93, size=30),
            'Study_Time_per_Week': np.random.uniform(1, 10, size=30).round(1),
            'Previous_Grade': np.random.randint(0, 101, size=30),
            'Gender': np.random.choice(['Male', 'Female'], size=30),
            'Extra_Courses': np.random.choice(['Yes', 'No'], size=30),
            'Internet_Access': np.random.choice(['Yes', 'No'], size=30),
            'Motivation_Level': np.random.choice(['Low', 'Medium', 'High', 'Very High'], size=30),
            'Parent_Education_Level': np.random.choice(['None', 'HighSchool', 'College', 'Masters', 'PhD'], size=30),
            'G3': np.random.randint(0, 101, size=30),
            'Medu': np.random.randint(0, 5, size=30), 'Fedu': np.random.randint(0, 5, size=30),
            'failures': np.random.randint(0, 4, size=30), 'schoolsup': np.random.choice(['yes', 'no'], size=30),
            'famsup': np.random.choice(['yes', 'no'], size=30), 'activities': np.random.choice(['yes', 'no'], size=30),
            'higher': np.random.choice(['yes', 'no'], size=30), 'romantic': np.random.choice(['yes', 'no'], size=30)
        }
        df = pd.DataFrame(dummy_data)
        # Minimal missing value handling for the dummy test data (consistent with data_processing.py)
        cols_for_nan = ['Age', 'Previous_Grade', 'Gender', 'Motivation_Level', 'Parent_Education_Level', 'Medu', 'failures']
        for col in cols_for_nan:
            if col in df.columns:
                idx = df.sample(frac=0.05, random_state=RANDOM_STATE).index
                df.loc[idx, col] = np.nan
                if pd.api.types.is_numeric_dtype(df[col]): df[col].fillna(df[col].median(), inplace=True)
                else: df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {data_path}")
        # Basic missing value imputation (as in data_processing.py)
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
        print("Applied basic missing value imputation to loaded data.")

    # Create target variable 'passed'
    df = create_target_variable(df, target_col_name='passed', grade_col_name='G3', threshold=50)
    if 'passed' not in df.columns:
        raise ValueError("Target column 'passed' could not be created or found in evaluation data.")

    y = df['passed']
    X_raw = df.drop(columns=['passed', 'G3'], errors='ignore')

    # Split data to get the test set, ensuring consistency with training split
    # We only need X_test and y_test here. The X_train_raw, y_train_raw are effectively discarded for evaluation.
    # However, train_test_split needs full X and y to ensure the split is correct.
    # If X_raw is small (e.g. from dummy data of size 30), test set might be very small.
    if len(X_raw) < 4 : # Minimum samples for stratify if test_size is 0.25
         print(f"Warning: Dataset too small for train/test split ({len(X_raw)} samples). Using entire dataset as test set for evaluation.")
         X_test = X_raw
         y_test = y
    else:
        _, X_test, _, y_test = train_test_split(X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"X_test raw shape (before interactions): {X_test.shape}, y_test shape: {y_test.shape}")

    # Apply interaction features to X_test
    X_test = add_interaction_features(X_test.copy())
    print(f"X_test shape after interactions: {X_test.shape}")

    # Preprocessing artifacts are no longer loaded individually.
    # The full pipeline will handle transformations.

    # 2. Define pipeline paths (updated from model_paths)
    pipeline_paths = {
        "Logistic Regression": "models/pipeline_logistic_regression.pkl",
        "Decision Tree": "models/pipeline_decision_tree.pkl",
        "Random Forest": "models/pipeline_random_forest.pkl"
    }

    all_metrics = []
    feature_importances_dfs = {}

    # 3. Load pipelines and evaluate
    for model_name, pipeline_path in pipeline_paths.items():
        print(f"\nEvaluating {model_name} pipeline...")
        pipeline = load_pipeline(pipeline_path) # Changed from load_model to load_pipeline

        if pipeline is None:
            print(f"Skipping {model_name} due to missing pipeline file.")
            metrics = {
                'model': model_name, 'accuracy': np.nan, 'precision': np.nan,
                'recall': np.nan, 'f1_score': np.nan, 'roc_auc': np.nan
            }
            all_metrics.append(metrics)
            # Initialize empty DFs for feature importances if pipeline is missing
            if model_name == "Logistic Regression":
                feature_importances_dfs["lr_coefficients"] = pd.DataFrame()
            else: # Decision Tree, Random Forest
                feature_importances_dfs[f"{model_name.lower().replace(' ', '_')}_feature_importances"] = pd.DataFrame()
            continue

        # Make predictions using the loaded pipeline on raw X_test
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

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

        # Feature importances/coefficients using the new function
        importances_df = get_feature_importances_from_pipeline(pipeline, top_n=10)

        if not importances_df.empty:
            print(f"\nTop 10 features/coefficients for {model_name} (from pipeline):")
            print(importances_df)
            # Save feature importances
            importance_filename_suffix = "feature_importances" if "importance" in importances_df.columns else "coefficients"
            filename = f"reports/figures/{model_name.lower().replace(' ', '_')}_{importance_filename_suffix}.csv"

            # Store for summary printout
            if model_name == "Logistic Regression":
                 feature_importances_dfs["lr_coefficients"] = importances_df
            else:
                 feature_importances_dfs[f"{model_name.lower().replace(' ', '_')}_feature_importances"] = importances_df

            importances_df.to_csv(filename, index=False)
            print(f"Saved feature importances/coefficients to {filename}")
        else:
            print(f"No feature importances or coefficients available/extracted for {model_name}.")

    # 4. Save all metrics to a CSV file
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = "reports/figures/metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nAll model metrics saved to {metrics_csv_path}")

    print("\nModel evaluation complete.")
    return metrics_df, feature_importances_dfs

if __name__ == '__main__':
    # The main guard now needs to handle dummy pipeline creation if needed,
    # or be simplified if standalone execution of evaluate.py without prior train.py run
    # is not a primary use case for full dummy model generation.
    # For now, we'll assume train.py has run and created the pipeline files.
    # If pipeline files are missing, evaluate_models() will print messages and skip evaluation for those.

    print("Running evaluate.py as main script...")
    os.makedirs('models', exist_ok=True) # Ensure models dir exists
    os.makedirs('data', exist_ok=True) # Ensure data dir exists for dummy data creation if needed by evaluate_models
    os.makedirs('reports/figures', exist_ok=True) # Ensure reports/figures dir exists

    # Check if actual pipeline files exist. If not, the evaluate_models function will handle it.
    # We could add creation of dummy *pipeline* files here for more robust standalone testing,
    # but that would require defining dummy ColumnTransformers and Pipelines.
    # For this iteration, evaluate_models will try to load actual pipelines.

    metrics_summary, importances_summary = evaluate_models()

    print("\n--- Main Script Test Output (evaluate.py) ---")
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
