import numpy as np
from typing import Any, Dict, Optional, Tuple
import logging
from datetime import datetime
import joblib
import os
from .drift_detector import DriftDetector

class ContinuousTrainer:
    def __init__(
        self,
        model: Any,
        drift_detector: DriftDetector,
        model_save_path: str = 'models',
        retrain_threshold: float = 0.1,
        min_samples_retrain: int = 1000
    ):
        """
        Initialize the continuous trainer.
        
        Args:
            model: The model to be continuously trained
            drift_detector: Instance of DriftDetector for monitoring drift
            model_save_path: Directory to save model versions
            retrain_threshold: Performance drop threshold to trigger retraining
            min_samples_retrain: Minimum number of samples required for retraining
        """
        self.model = model
        self.drift_detector = drift_detector
        self.model_save_path = model_save_path
        self.retrain_threshold = retrain_threshold
        self.min_samples_retrain = min_samples_retrain
        self.logger = logging.getLogger(__name__)
        
        # Create model save directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
        # Initialize training history
        self.training_history = []
        
    def check_and_retrain(
        self,
        new_data: np.ndarray,
        new_labels: np.ndarray,
        current_predictions: Optional[np.ndarray] = None
    ) -> Tuple[bool, Dict]:
        """
        Check for drift and retrain if necessary.
        
        Args:
            new_data: New training data
            new_labels: New training labels
            current_predictions: Current model predictions on new data (optional)
            
        Returns:
            Tuple of (retrained, metrics)
        """
        metrics = {}
        
        # Check for data drift
        data_drift_detected, data_drift_metrics = self.drift_detector.detect_data_drift(new_data)
        metrics['data_drift'] = data_drift_metrics
        
        # Check for concept drift if predictions are provided
        concept_drift_detected = False
        if current_predictions is not None:
            concept_drift_detected, concept_drift_metrics = self.drift_detector.detect_concept_drift(
                current_predictions, new_labels
            )
            metrics['concept_drift'] = concept_drift_metrics
        
        # Determine if retraining is needed
        should_retrain = (
            (data_drift_detected or concept_drift_detected) and
            len(new_data) >= self.min_samples_retrain
        )
        
        if should_retrain:
            self.logger.info("Drift detected. Initiating model retraining...")
            
            # Retrain the model
            retrain_metrics = self._retrain_model(new_data, new_labels)
            metrics['retrain'] = retrain_metrics
            
            # Save the new model version
            self._save_model_version()
            
            # Update drift detector reference data
            self.drift_detector.update_reference_data(new_data)
            
            return True, metrics
        
        return False, metrics
    
    def _retrain_model(self, new_data: np.ndarray, new_labels: np.ndarray) -> Dict:
        """Retrain the model on new data."""
        try:
            # Record training start time
            start_time = datetime.now()
            
            # Fit the model
            self.model.fit(new_data, new_labels)
            
            # Calculate training metrics
            train_predictions = self.model.predict(new_data)
            train_accuracy = np.mean(train_predictions == new_labels)
            
            # Record training event
            training_event = {
                'timestamp': start_time,
                'samples_count': len(new_data),
                'training_accuracy': train_accuracy,
                'training_duration': (datetime.now() - start_time).total_seconds()
            }
            self.training_history.append(training_event)
            
            return training_event
            
        except Exception as e:
            self.logger.error(f"Error during model retraining: {str(e)}")
            return {'error': str(e)}
    
    def _save_model_version(self):
        """Save a new version of the model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_save_path, f'model_v{timestamp}.joblib')
        
        try:
            joblib.dump(self.model, model_path)
            self.logger.info(f"Model saved successfully at: {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def get_training_summary(self) -> Dict:
        """Get summary of training history."""
        if not self.training_history:
            return {'message': 'No training events recorded'}
            
        return {
            'total_retraining_events': len(self.training_history),
            'latest_training': self.training_history[-1],
            'average_accuracy': np.mean([event['training_accuracy'] for event in self.training_history]),
            'average_duration': np.mean([event['training_duration'] for event in self.training_history])
        } 