import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
RUNS_DIR = "runs"  # Directory to store run data


def train_model(X_train, y_train, model_params, run_id):
    """
    Trains a Logistic Regression model with the given parameters.  Saves model & params.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_params (dict): Parameters for Logistic Regression.
        run_id (str): Unique identifier for the run.

    Returns:
        LogisticRegression: Trained model.
    """
    # Create run directory
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save parameters
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(model_params, f)

    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    # Save the model using pickle
    with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Trained model and saved to {run_dir}")
    return model



def evaluate_model(model, X_test, y_test, run_id):
    """
    Evaluates the trained model on the test set.  Saves metrics.

    Args:
        model (LogisticRegression): Trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        run_id (str): Unique identifier for the run.

    Returns:
        dict: Evaluation metrics (accuracy, classification report, confusion matrix).
    """
    run_dir = os.path.join(RUNS_DIR, run_id)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred).tolist()

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": confusion,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    logger.info(f"Evaluated model and saved metrics to {run_dir}")
    return metrics



def tune_model(X_train, y_train, tuning_strategy, run_id, c_values, solvers, penalties):
    """
    Tunes the Logistic Regression model using GridSearchCV or RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        tuning_strategy (str): "grid" for GridSearchCV or "random" for RandomizedSearchCV.
        run_id (str): Unique identifier for the run.
        c_values (list): List of C values for tuning.
        solvers (list): List of solvers for tuning.
        penalties (list): List of penalties for tuning.

    Returns:
        LogisticRegression: Best trained model.
        dict: Results.
    """
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    param_grid = {
        "C": c_values,
        "solver": solvers,
        "penalty": penalties,
    }
    if tuning_strategy == "grid":
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42), param_grid, cv=3, verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        results = {
            "best_params": grid_search.best_params_,
            "best_accuracy": grid_search.best_score_,
        }
        # Save results
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f)
        #save model
        with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
        logger.info(f"Grid search complete.  Best model and results saved to {run_dir}")
        return best_model, results

    elif tuning_strategy == "random":
        random_search = RandomizedSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            n_iter=10,
            cv=3,
            verbose=1,
            random_state=42,
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        results = {
            "best_params": random_search.best_params_,
            "best_accuracy": random_search.best_score_,
        }
        # Save results.
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f)
        # Save model
        with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
            pickle.dump(best_model, f)

        logger.info(
            f"Randomized search complete.  Best model and results saved to {run_dir}"
        )
        return best_model, results
    else:
        raise ValueError("Invalid tuning strategy. Choose 'grid' or 'random'.")



def load_model(run_id):
    """
    Loads the trained model from the specified run.

    Args:
        run_id (str): Unique identifier for the run.

    Returns:
        LogisticRegression: The loaded model.
    """
    run_dir = os.path.join(RUNS_DIR, run_id)
    model_path = os.path.join(run_dir, "model.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {e}")
