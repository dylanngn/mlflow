import fire
import logging
import os
from sklearn.model_selection import train_test_split

# Import functions from utils
from utils.data_utils import create_training_data
from utils.model_utils import train_model, tune_model, evaluate_model, load_model
from utils.report_utils import generate_report

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
EXPERIMENT_NAME = "Classification_Model_Experiment"
RUNS_DIR = "runs"  # Directory to store run data

class MLWorkflow:
    """
    A command-line interface for the ML workflow.
    """

    def __init__(self):
        """
        Initializes the MLWorkflow class.
        """
        # Ensure the runs directory exists
        os.makedirs(RUNS_DIR, exist_ok=True)

    def train(
        self,
        tuning_strategy: str = "none",
        c_values: str = "0.01,0.1,1,10,100",
        solvers: str = "liblinear,saga",
        penalties: str = "l1,l2",
    ):
        """
        Trains and tunes a classification model.

        Args:
            tuning_strategy (str, optional): Tuning strategy ('none', 'grid', 'random'). Defaults to 'none'.
            c_values (str, optional): Comma-separated values for the 'C' parameter. Defaults to '0.01,0.1,1,10,100'.
            solvers (str, optional): Comma-separated values for the 'solver' parameter. Defaults to 'liblinear,saga'.
            penalties (str, optional): Comma-separated values for the 'penalty' parameter. Defaults to 'l1,l2'.
        """
        logger.info(f"Starting training with tuning strategy: {tuning_strategy}")

        # Create and split data
        X, y = create_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if tuning_strategy == "none":
            # --- Base Model Training ---
            base_model_params = {
                "C": 1,
                "solver": "liblinear",
                "penalty": "l2",
                "random_state": 42,
            }
            base_run_id = "base_model_run"
            train_model(X_train, y_train, base_model_params, base_run_id)
            model = load_model(base_run_id)
            evaluate_model(model, X_test, y_test, base_run_id)
            logger.info("Base model training complete.")

        elif tuning_strategy in ["grid", "random"]:
            # --- Tuning ---
            c_values_list = [float(c) for c in c_values.split(",")]
            solvers_list = solvers.split(",")
            penalties_list = penalties.split(",")

            tuned_model, _ = tune_model(
                X_train,
                y_train,
                tuning_strategy,
                "tuned_model_run",  # Fix the run_id
                c_values_list,
                solvers_list,
                penalties_list
            )
            model = load_model("tuned_model_run")
            evaluate_model(model, X_test, y_test, "tuned_model_run")
            logger.info(f"{tuning_strategy.capitalize()} search complete.")
        else:
            raise ValueError(
                "Invalid tuning strategy.  Choose 'none', 'grid', or 'random'."
            )
        return model

    def report(self):
        """
        Generates a report comparing the performance of different model runs.
        """
        logger.info("Generating model comparison report...")
        generate_report()
        logger.info("Report generation complete.")

    def run_app(self):
        """
        Runs the Flask application.
        """
        logger.info("Starting the Flask application...")
        from app.app import create_app_instance # changed from create_app

        app = create_app_instance() # changed from create_app()
        app.run(debug=True, port=5000)  # Keep debug True for development

if __name__ == "__main__":
    fire.Fire(MLWorkflow)
