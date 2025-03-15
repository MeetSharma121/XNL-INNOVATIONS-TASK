import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import wasserstein_distance

class DriftDetector:
    def __init__(self, reference_data: np.ndarray, window_size: int = 1000, threshold: float = 0.05):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Initial data distribution to compare against
            window_size: Size of the sliding window for drift detection
            threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.window_size = window_size
        self.threshold = threshold
        self.scaler = StandardScaler().fit(reference_data)
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring metrics
        self.drift_history = []
        self.performance_history = []
        
    def detect_data_drift(self, new_data: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect data drift using statistical tests.
        
        Args:
            new_data: New batch of data to compare against reference
            
        Returns:
            Tuple of (drift_detected, drift_metrics)
        """
        if len(new_data) < self.window_size:
            self.logger.warning("New data batch smaller than window size")
            return False, {}
            
        # Standardize both datasets
        ref_scaled = self.scaler.transform(self.reference_data)
        new_scaled = self.scaler.transform(new_data)
        
        drift_metrics = {}
        
        # Perform Kolmogorov-Smirnov test for each feature
        drift_detected = False
        for feature_idx in range(ref_scaled.shape[1]):
            ks_statistic, p_value = stats.ks_2samp(
                ref_scaled[:, feature_idx],
                new_scaled[:, feature_idx]
            )
            
            # Calculate Wasserstein distance
            w_distance = wasserstein_distance(
                ref_scaled[:, feature_idx],
                new_scaled[:, feature_idx]
            )
            
            drift_metrics[f'feature_{feature_idx}'] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'wasserstein_distance': w_distance
            }
            
            if p_value < self.threshold:
                drift_detected = True
                
        # Record drift event
        self.drift_history.append({
            'timestamp': datetime.now(),
            'drift_detected': drift_detected,
            'metrics': drift_metrics
        })
        
        return drift_detected, drift_metrics
        
    def detect_concept_drift(self, predictions: np.ndarray, true_labels: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect concept drift by monitoring model performance.
        
        Args:
            predictions: Model predictions
            true_labels: Actual labels
            
        Returns:
            Tuple of (drift_detected, performance_metrics)
        """
        # Calculate performance metrics
        accuracy = np.mean(predictions == true_labels)
        
        # Add to performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'accuracy': accuracy
        })
        
        # Detect significant drop in performance
        if len(self.performance_history) >= 2:
            current_perf = accuracy
            historical_perf = pd.DataFrame(self.performance_history[:-1])['accuracy'].mean()
            
            performance_drop = historical_perf - current_perf
            concept_drift_detected = performance_drop > self.threshold
            
            metrics = {
                'current_performance': current_perf,
                'historical_performance': historical_perf,
                'performance_drop': performance_drop
            }
            
            return concept_drift_detected, metrics
            
        return False, {'message': 'Not enough historical data'}
        
    def get_drift_summary(self) -> Dict:
        """Get summary of drift detection history."""
        return {
            'total_checks': len(self.drift_history),
            'drift_events': sum(1 for event in self.drift_history if event['drift_detected']),
            'latest_drift_metrics': self.drift_history[-1] if self.drift_history else None,
            'performance_trend': pd.DataFrame(self.performance_history).to_dict() if self.performance_history else None
        }
        
    def update_reference_data(self, new_reference_data: np.ndarray):
        """Update the reference data and recalibrate the scaler."""
        self.reference_data = new_reference_data
        self.scaler = StandardScaler().fit(new_reference_data)
        self.logger.info("Reference data and scaler updated") 