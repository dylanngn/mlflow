import logging
import os
import sys
from datetime import datetime
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

def main():
    """
    Main function to run the ML workflow.  Handles command-line arguments directly.
    """
    # Ensure the runs directory exists
    os.makedirs(RUNS_DIR, exist_ok=True)
    sys.path.append(os.getcwd())

    if len(sys.argv) < 2:
        print("Usage: python run.py <command> [options]")
        print("Commands: train, report, run_app")
        return

    command = sys.argv[1]

    if command == "train":
        tuning_strategy = "none"
        c_values = "0.01,0.1,1,10,100"
        solvers = "liblinear,saga"
        penalties = "l1,l2"

        if len(sys.argv) > 2:
            for arg in sys.argv[2:]:
                if "=" in arg:
                    key, value = arg.split("=")
                    if key == "tuning_strategy":
                        tuning_strategy = value
                    elif key == "c_values":
                        c_values = value
                    elif key == "solvers":
                        solvers = value
                    elif key == "penalties":
                        penalties = value

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"base_model_run_{timestamp}"
            train_model(X_train, y_train, base_model_params, run_id)
            model = load_model(run_id)
            evaluate_model(model, X_test, y_test, run_id)
            logger.info(f"Base model training complete. Run ID: {run_id}")

        elif tuning_strategy in ["grid", "random"]:
            # --- Tuning ---
            c_values_list = [float(c) for c in c_values.split(",")]
            solvers_list = solvers.split(",")
            penalties_list = penalties.split(",")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{tuning_strategy}_search_run_{timestamp}"
            tuned_model, _ = tune_model(
                X_train,
                y_train,
                tuning_strategy,
                run_id,
                c_values_list,
                solvers_list,
                penalties_list
            )
            model = load_model(run_id)
            evaluate_model(model, X_test, y_test, run_id)
            logger.info(f"{tuning_strategy.capitalize()} search complete. Run ID: {run_id}")
        else:
            raise ValueError(
                "Invalid tuning strategy.  Choose 'none', 'grid', or 'random'."
            )
        return model

    elif command == "report":
        logger.info("Generating model comparison report...")
        generate_report()
        logger.info("Report generation complete.")

    elif command == "run_app":
        logger.info("Starting the Flask application...")
        from app.app import create_app_instance

        app = create_app_instance()
        app.run(debug=True, port=5000)
    else:
        print("Error: Invalid command")
        print("Usage: python run.py <command> [options]")
        print("Commands: train, report, run_app")
        return


if __name__ == "__main__":
    main()
