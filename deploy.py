import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import os
from datetime import datetime
from testing.model_evaluation.evaluator import ModelEvaluator
from testing.monitoring.drift_detector import DriftDetector
from testing.monitoring.continuous_trainer import ContinuousTrainer
from testing.test_suites.custom_test_suite import CustomTestSuite, TestCase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000, n_features: int = 10):
    """Generate sample data for demonstration."""
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple binary classification
    return X, y

def generate_drift_data(n_samples: int = 1000, n_features: int = 10, drift_factor: float = 2.0):
    """Generate data with drift for testing."""
    X = np.random.randn(n_samples, n_features) * drift_factor
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def setup_directories():
    """Create necessary directories."""
    dirs = ['models', 'results', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"Created directory: {dir_name}")

def main():
    logger.info("Starting deployment of testing and monitoring framework...")
    
    # Setup directories
    setup_directories()
    
    # Generate initial data
    X_train, y_train = generate_sample_data(n_samples=1000)
    X_val, y_val = generate_sample_data(n_samples=200)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Initial model training completed")
    
    # Initialize components
    evaluator = ModelEvaluator(model, k_folds=5)
    drift_detector = DriftDetector(X_train, window_size=100)
    trainer = ContinuousTrainer(
        model=model,
        drift_detector=drift_detector,
        model_save_path='models',
        retrain_threshold=0.1
    )
    test_suite = CustomTestSuite(model, evaluator)
    
    # 1. Model Evaluation
    logger.info("Running initial model evaluation...")
    cv_metrics = evaluator.perform_cross_validation(X_train, y_train)
    logger.info(f"Cross-validation metrics: {cv_metrics}")
    
    # 2. Add test cases
    logger.info("Setting up test cases...")
    # Add standard test cases
    test_suite.add_edge_case(
        X_val[0],
        y_val[0],
        "Sample edge case for binary classification"
    )
    
    # Add conversation test case (if model supports it)
    try:
        test_suite.add_conversation_test(
            ["What is machine learning?"],
            ["Machine learning is a subset of artificial intelligence..."]
        )
    except:
        logger.warning("Skipping conversation tests - model doesn't support text generation")
    
    # 3. Run test suite
    logger.info("Running test suite...")
    test_results = test_suite.run_tests()
    test_suite.save_results("results/test_results.json")
    logger.info(f"Test results: {test_results}")
    
    # 4. Simulate drift detection and continuous training
    logger.info("Simulating data drift...")
    X_drift, y_drift = generate_drift_data(n_samples=500)
    drift_predictions = model.predict(X_drift)
    
    retrained, metrics = trainer.check_and_retrain(
        X_drift,
        y_drift,
        current_predictions=drift_predictions
    )
    
    if retrained:
        logger.info("Model retrained due to drift")
        logger.info(f"Retraining metrics: {metrics}")
    else:
        logger.info("No significant drift detected")
    
    # 5. Get monitoring summaries
    drift_summary = drift_detector.get_drift_summary()
    training_summary = trainer.get_training_summary()
    test_summary = test_suite.get_test_summary()
    
    logger.info("=== Final Summaries ===")
    logger.info(f"Drift Summary: {drift_summary}")
    logger.info(f"Training Summary: {training_summary}")
    logger.info(f"Test Summary: {test_summary}")
    
    logger.info("Deployment completed successfully")

if __name__ == "__main__":
    main() 