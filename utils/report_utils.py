import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_report():
    """
    Generates a report comparing the performance of different model runs.
    The report is saved to a JSON file.
    """
    runs_dir = "runs"  # Directory where run data is stored
    report_data = {}

    # Get all subdirectories (which represent runs) in the runs directory
    run_ids = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]

    if not run_ids:
        logger.warning("No runs found to generate report.")
        return

    for run_id in run_ids:
        run_dir = os.path.join(runs_dir, run_id)
        metrics_file = os.path.join(run_dir, "metrics.json")
        params_file = os.path.join(run_dir, "params.json")
        results_file = os.path.join(run_dir, "results.json") # add this

        # Load metrics and parameters, handling potential file errors
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            with open(params_file, "r") as f:
                params = json.load(f)
            #load results
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
            except FileNotFoundError:
                results = {}

        except FileNotFoundError:
            logger.warning(f"Metrics or parameters file not found for run {run_id}. Skipping.")
            continue  # Skip to the next run

        # Extract relevant metrics and parameters
        accuracy = metrics.get("accuracy")
        classification_report = metrics.get("classification_report")
        confusion_matrix = metrics.get("confusion_matrix")
        model_params = params
        best_params = results.get('best_params', {}) # add this
        best_accuracy = results.get('best_accuracy', None) # add this

        # Store the extracted information
        report_data[run_id] = {
            "accuracy": accuracy,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix,
            "model_params": model_params,
            "best_params": best_params, # add this
            "best_accuracy": best_accuracy
        }

    # Find the best run based on accuracy.
    best_run_id = None
    best_accuracy = -1
    for run_id, data in report_data.items():
        if data["accuracy"] is not None and data["accuracy"] > best_accuracy:
            best_accuracy = data["accuracy"]
            best_run_id = run_id

    report_data["best_run_id"] = best_run_id

    # Save the report to a JSON file
    report_file = "model_comparison_report.json"
    try:
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated and saved to {report_file}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        raise
    return report_data
