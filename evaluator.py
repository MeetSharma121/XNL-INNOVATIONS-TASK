import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Any
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import logging

class ModelEvaluator:
    def __init__(self, model: Any, k_folds: int = 5):
        """
        Initialize the model evaluator.
        
        Args:
            model: The model to evaluate
            k_folds: Number of folds for cross-validation
        """
        self.model = model
        self.k_folds = k_folds
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK and Rouge scorer
        try:
            nltk.download('punkt')
        except:
            self.logger.warning("NLTK punkt download failed. BLEU score might not work.")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate standard classification metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def calculate_nlp_metrics(self, references: List[str], predictions: List[str]) -> Dict:
        """Calculate NLP-specific metrics (BLEU, ROUGE)."""
        # Calculate BLEU score
        bleu_scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = nltk.word_tokenize(ref)
            pred_tokens = nltk.word_tokenize(pred)
            bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens))
        
        # Calculate ROUGE scores
        rouge_scores = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }
        
        for ref, pred in zip(references, predictions):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rouge2'] += scores['rougeL'].fmeasure
            
        # Average ROUGE scores
        for key in rouge_scores:
            rouge_scores[key] /= len(references)
            
        return {
            'bleu': np.mean(bleu_scores),
            **rouge_scores
        }

    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform k-fold cross-validation."""
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = self.model.predict(X_val)
            
            # Calculate metrics
            metrics = self.calculate_classification_metrics(y_val, y_pred)
            fold_metrics.append(metrics)
            
            self.logger.info(f"Fold {fold + 1} metrics: {metrics}")
        
        # Calculate average metrics across folds
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
            avg_metrics[f"{metric}_std"] = np.std([fold[metric] for fold in fold_metrics])
        
        return avg_metrics

    def evaluate_edge_cases(self, edge_cases: List[Tuple[Any, Any]]) -> Dict:
        """Evaluate model performance on edge cases."""
        X_edge, y_edge = zip(*edge_cases)
        X_edge = np.array(X_edge)
        y_edge = np.array(y_edge)
        
        y_pred = self.model.predict(X_edge)
        metrics = self.calculate_classification_metrics(y_edge, y_pred)
        
        return {
            'edge_case_' + k: v for k, v in metrics.items()
        } 