from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_logistic_regression(random_state=42, C=1.0, max_iter=1000):
    """
    Returns an untrained Logistic Regression model.

    Parameters:
    ----------
    random_state : int, default=42
        Random state for reproducibility.
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
    max_iter : int, default=1000
        Maximum number of iterations taken for the solvers to converge.

    Returns:
    -------
    sklearn.linear_model.LogisticRegression
        Untrained Logistic Regression estimator.
    """
    return LogisticRegression(
        penalty='l2',
        C=C,
        solver='liblinear',
        random_state=random_state,
        max_iter=max_iter
    )

def get_decision_tree(random_state=42, max_depth=None, min_samples_split=2):
    """
    Returns an untrained Decision Tree Classifier model.

    Parameters:
    ----------
    random_state : int, default=42
        Random state for reproducibility.
    max_depth : int or None, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    Returns:
    -------
    sklearn.tree.DecisionTreeClassifier
        Untrained Decision Tree Classifier estimator.
    """
    return DecisionTreeClassifier(
        criterion='gini',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

def get_random_forest(random_state=42, n_estimators=100, max_depth=None, n_jobs=-1):
    """
    Returns an untrained Random Forest Classifier model.

    Parameters:
    ----------
    random_state : int, default=42
        Random state for reproducibility.
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int or None, default=None
        The maximum depth of the trees. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.

    Returns:
    -------
    sklearn.ensemble.RandomForestClassifier
        Untrained Random Forest Classifier estimator.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion='gini',
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )

if __name__ == '__main__':
    # Basic test to ensure functions can be called and return model instances
    print("Running models.py as main script...")

    lr_model = get_logistic_regression()
    print(f"Logistic Regression model: {lr_model}")

    dt_model = get_decision_tree()
    print(f"Decision Tree model: {dt_model}")

    rf_model = get_random_forest()
    print(f"Random Forest model: {rf_model}")

    print("\nSuccessfully retrieved model instances.")
    print("models.py script run complete.")
