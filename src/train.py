import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Assuming data_processing.py and models.py are in the same directory (src)
# and the script is run from the project root or src directory is in PYTHONPATH.
# For robust imports, especially if running from project root:
try:
    from src.data_processing import preprocess_data
    from src.models import get_logistic_regression, get_decision_tree, get_random_forest
except ModuleNotFoundError:
    # Fallback for cases where src is not in python path (e.g. running script directly from src)
    from data_processing import preprocess_data
    from models import get_logistic_regression, get_decision_tree, get_random_forest


def train_models():
    """
    Trains and saves Logistic Regression, Decision Tree, and Random Forest models
    using GridSearchCV.
    """
    print("Starting model training process...")

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    # Ensure data directory exists for preprocess_data, though it creates dummy if not found
    os.makedirs('data', exist_ok=True)

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    # It will use/create 'data/student_data.csv' as defined in preprocess_data
    # If student_data.csv is not found, preprocess_data will create and use a dummy one.
    # Ensure a dummy one is created if not present for train.py to run.
    dummy_data_path = 'data/student_data.csv'
    if not os.path.exists(dummy_data_path):
        print(f"'{dummy_data_path}' not found. data_processing.py will create a dummy CSV for training.")
        # The data_processing.py script's main guard or preprocess_data itself handles dummy creation.
        # We just need to ensure it's called.

    X_train, X_test, y_train, y_test = preprocess_data(df_path=dummy_data_path)
    print("Data loading and preprocessing complete.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 2. Define models and hyperparameter grids
    models_to_train = {
        "Logistic Regression": {
            "estimator": get_logistic_regression(random_state=42),
            "params": {'C': [0.01, 0.1, 1.0, 10, 100]}
        },
        "Decision Tree": {
            "estimator": get_decision_tree(random_state=42),
            "params": {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}
        },
        "Random Forest": {
            "estimator": get_random_forest(random_state=42),
            "params": {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20]}
        }
    }

    best_estimators = {}

    # 3. Train models using GridSearchCV
    for model_name, model_info in models_to_train.items():
        print(f"\nTraining {model_name}...")
        grid_search = GridSearchCV(
            estimator=model_info["estimator"],
            param_grid=model_info["params"],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1 # Adds some verbosity to GridSearchCV
        )
        grid_search.fit(X_train, y_train)

        best_estimators[model_name] = grid_search.best_estimator_

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validated accuracy for {model_name}: {grid_search.best_score_:.4f}")

        # 4. Serialize the best estimator
        model_filename = f"models/best_{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(grid_search.best_estimator_, model_filename)
        print(f"Saved best {model_name} to {model_filename}")

    print("\nModel training and saving complete.")
    return best_estimators

if __name__ == '__main__':
    print("Running train.py as main script...")
    # The train_models function handles data loading (including dummy data if necessary)
    # and model training.
    trained_models = train_models()

    print("\n--- Main Script Test Output ---")
    if trained_models:
        print("The following models were trained and saved:")
        for name, model in trained_models.items():
            print(f"- {name}: {model}")
    else:
        print("No models were trained.")
    print("train.py script run complete.")
