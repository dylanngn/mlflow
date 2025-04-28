import pandas as pd
from flask import Flask, request, jsonify
import logging
import os
import pickle  # For saving and loading models

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variable to hold the best model
best_model = None
EXPERIMENT_CONFIG_FILE = "experiment_config.json"  # File to store experiment config
RUNS_DIR = "runs"


def create_app():
    """
    Creates a Flask application to serve the best trained model.

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        Flask: Flask application instance.
    """
    app = Flask(__name__)
    # Add a logger for the flask app
    flask_logger = logging.getLogger("werkzeug")
    flask_logger.setLevel(logging.INFO)
    app.logger.addHandler(flask_logger)

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Endpoint for making predictions using the best trained model.
        Expects a JSON payload with the features.
        """
        try:
            data = request.get_json()
            app.logger.info(f"Received data for prediction: {data}")  # Log the received data

            # Basic validation of input data
            if not data or not isinstance(data, dict):
                return jsonify({"error": "Invalid input. Expected a JSON object."}), 400

            # Convert input data to DataFrame
            try:
                input_df = pd.DataFrame([data])
            except Exception as e:
                error_message = f"Error converting input to DataFrame: {e}"
                app.logger.error(error_message)
                return jsonify({"error": error_message}), 400

            # Use the pre-loaded model for prediction
            prediction = best_model.predict(input_df)
            app.logger.info(f"Prediction: {prediction}")
            return jsonify({"prediction": int(prediction[0])}), 200

        except Exception as e:
            error_message = f"An error occurred during prediction: {e}"
            app.logger.error(error_message)
            return jsonify({"error": error_message}), 500

    @app.route("/health", methods=["GET"])
    def health_check():
        """
        Endpoint for health check.
        """
        return jsonify({"status": "ok"}), 200

    # Load the best model globally when the app starts.
    # This ensures the model is loaded only once, not on every prediction request.
    global best_model
    try:
        experiment_config = load_experiment_config()  # Load experiment config
        if experiment_config and "best_run_id" in experiment_config:
            best_run_id = experiment_config["best_run_id"]
            best_model = load_model(best_run_id)  # Load using the utility function
            app.logger.info(f"Loaded best model from run ID: {best_run_id}")
        else:
            app.logger.warning(
                "No best run found.  The application may not function correctly until a model is trained."
            )
            best_model = None
    except Exception as e:
        app.logger.error(f"Error loading best model: {e}")
        best_model = None

    return app


def load_experiment_config():
    """
    Loads experiment-level configuration data from a JSON file.

    Returns:
        dict: The configuration data, or an empty dict if not found.
    """
    config_file_path = EXPERIMENT_CONFIG_FILE
    try:
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                config = json.load(f)
            return config
        else:
            return {}  # Return empty dict if file doesn't exist
    except Exception as e:
        logger.error(f"Failed to load experiment config: {e}")
        return {}  # Return empty dict on error, so app can run


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
