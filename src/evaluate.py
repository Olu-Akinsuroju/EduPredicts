import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Assuming data_processing.py is in the same directory (src)
# and the script is run from the project root or src directory is in PYTHONPATH.
try:
    from src.data_processing import preprocess_data
except ModuleNotFoundError:
    from data_processing import preprocess_data


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
    os.makedirs('data', exist_ok=True) # For preprocess_data

    # 1. Load test data
    print("Loading and preprocessing data for evaluation...")
    # It will use/create 'data/student_data.csv' as defined in preprocess_data
    dummy_data_path = 'data/student_data.csv'
    if not os.path.exists(dummy_data_path):
        print(f"'{dummy_data_path}' not found. data_processing.py will create a dummy CSV for evaluation.")
        # data_processing.py handles dummy creation if file is missing.

    X_train, X_test, y_train, y_test = preprocess_data(df_path=dummy_data_path)
    print("Data loading and preprocessing complete for evaluation.")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

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
