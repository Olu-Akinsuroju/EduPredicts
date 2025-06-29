import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Assuming data_processing.py and models.py are in the same directory (src)
# and the script is run from the project root or src directory is in PYTHONPATH.
try:
    from src.data_processing import preprocess_data
    from src.models import get_logistic_regression, get_decision_tree, get_random_forest
except ModuleNotFoundError:
    from data_processing import preprocess_data
    from models import get_logistic_regression, get_decision_tree, get_random_forest


def train_models():
    """
    Trains and saves Logistic Regression, Decision Tree, and Random Forest models
    using GridSearchCV with a ColumnTransformer preprocessing pipeline.
    """
    print("Starting model training process...")

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 1. Load raw data (preprocess_data now returns raw X_train, X_test)
    print("Loading raw data...")
    dummy_data_path = 'data/student_data.csv' # This path is used by preprocess_data
    X_train, X_test, y_train, y_test = preprocess_data(df_path=dummy_data_path)
    print("Raw data loading complete.")
    print(f"X_train raw shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 2. Define feature lists (based on user spec and dummy data from data_processing.py)
    # These names must match columns in X_train
    numeric_features = ['Age', 'Absences', 'Study_Time_per_Week', 'Previous_Grade', 'Medu', 'Fedu', 'failures']
    # Binary features will be handled by OneHotEncoder with drop='first' or similar
    # OneHotFeatures includes both binary and multi-category ones from user spec + dummy data
    categorical_features = [
        'Gender', 'Extra_Courses', 'Internet_Access', # User-specified binary
        'Motivation_Level', 'Parent_Education_Level', # User-specified OHE
        'schoolsup', 'famsup', 'activities', 'higher', 'romantic' # Others from dummy
    ]

    # Filter lists to only include columns present in X_train to avoid errors if dummy data changes
    # or if real data is used and has a different schema.
    # This is important because ColumnTransformer will fail if a specified column is missing.
    actual_numeric_features = [col for col in numeric_features if col in X_train.columns]
    actual_categorical_features = [col for col in categorical_features if col in X_train.columns]

    print(f"Actual numeric features to be used: {actual_numeric_features}")
    print(f"Actual categorical features to be used: {actual_categorical_features}")


    # 3. Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    # 4. Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, actual_numeric_features),
            ('cat', categorical_transformer, actual_categorical_features)
        ],
        remainder='drop' # Drop other columns not specified
    )

    # 5. Define models and hyperparameter grids for GridSearchCV
    # Parameters now need to be prefixed with 'classifier__'
    models_to_train = {
        "Logistic Regression": {
            "estimator": get_logistic_regression(random_state=42),
            "params": {'classifier__C': [0.01, 0.1, 1.0, 10, 100]}
        },
        "Decision Tree": {
            "estimator": get_decision_tree(random_state=42),
            "params": {'classifier__max_depth': [None, 5, 10, 20],
                       'classifier__min_samples_split': [2, 5, 10]}
        },
        "Random Forest": {
            "estimator": get_random_forest(random_state=42),
            "params": {'classifier__n_estimators': [50, 100, 200],
                       'classifier__max_depth': [None, 5, 10, 20]}
        }
    }

    best_pipelines = {}

    # 6. Train models using GridSearchCV with the full pipeline
    for model_name, model_config in models_to_train.items():
        print(f"\nTraining {model_name} with preprocessing pipeline...")

        # Create the full pipeline: preprocessor + classifier
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_config["estimator"])
        ])

        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=model_config["params"],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_pipelines[model_name] = grid_search.best_estimator_ # This is the best *fitted pipeline*

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validated accuracy for {model_name} (pipeline): {grid_search.best_score_:.4f}")

        # 7. Serialize the best *pipeline*
        pipeline_filename = f"models/pipeline_{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(grid_search.best_estimator_, pipeline_filename)
        print(f"Saved best {model_name} pipeline to {pipeline_filename}")

    print("\nModel training and pipeline saving complete.")
    return best_pipelines

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
