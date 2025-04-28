import pandas as pd
from sklearn.datasets import make_classification
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_training_data():
    """
    Generates synthetic classification data.

    Returns:
        pd.DataFrame: Feature matrix (X).
        pd.Series: Target vector (y).
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        class_sep=0.8,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y, name="target")
    logger.info("Generated training data")
    return X, y
