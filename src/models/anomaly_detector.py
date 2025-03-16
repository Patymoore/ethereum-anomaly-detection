import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from pyod.models.auto_encoder import AutoEncoder

class AnomalyDetector:
    """
    Class for detecting anomalies in Ethereum transactions.
    """
    
    def __init__(self, method: str = 'isolation_forest', **kwargs):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        method : str, default='isolation_forest'
            Method to use for anomaly detection.
            Options: 'isolation_forest', 'one_class_svm', 'lof', 'autoencoder', 'xgboost'
        **kwargs : dict
            Additional parameters for the selected method.
        """
        self.method = method
        self.model = None
        self.kwargs = kwargs
        
        # Initialize the model based on the method
        if method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=kwargs.get('contamination', 0.05),
                random_state=kwargs.get('random_state', 42),
                n_estimators=kwargs.get('n_estimators', 100),
                max_samples=kwargs.get('max_samples', 'auto')
            )
        elif method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=kwargs.get('nu', 0.05),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale')
            )
        elif method == 'lof':
            self.model = LocalOutlierFactor(
                n_neighbors=kwargs.get('n_neighbors', 20),
                contamination=kwargs.get('contamination', 0.05),
                novelty=True  # Required for prediction on new data
            )
        elif method == 'autoencoder':
            self.model = AutoEncoder(
                hidden_neurons=kwargs.get('hidden_neurons', [64, 32, 32, 64]),
                contamination=kwargs.get('contamination', 0.05),
                epochs=kwargs.get('epochs', 100),
                batch_size=kwargs.get('batch_size', 32)
            )
        elif method == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 3),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AnomalyDetector':
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for training.
        y : pd.Series, optional
            Target variable (required for supervised methods like XGBoost).
            
        Returns:
        --------
        self
        """
        if self.method == 'xgboost':
            if y is None:
                raise ValueError("Target variable (y) is required for XGBoost")
            self.model.fit(X, y)
        else:
            # Unsupervised methods
            self.model.fit(X)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction.
            
        Returns:
        --------
        np.ndarray
            Predictions (1 for normal, -1 for anomaly) for unsupervised methods,
            or binary predictions (0, 1) for supervised methods.
        """
        if self.method == 'xgboost':
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of anomalies (for methods that support it).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction.
            
        Returns:
        --------
        np.ndarray
            Probability scores.
        """
        if self.method == 'xgboost':
            return self.model.predict_proba(X)[:, 1]
        elif self.method == 'isolation_forest':
            # Convert decision function to probability-like score
            scores = self.model.decision_function(X)
            # Normalize to [0, 1] range where 1 is anomaly
            return 0.5 - scores / (2 * max(abs(scores.min()), abs(scores.max())))
        elif self.method == 'one_class_svm':
            # Similar approach for SVM
            scores = self.model.decision_function(X)
            return 1 - 1 / (1 + np.exp(-scores))
        elif self.method == 'autoencoder':
            return self.model.predict_proba(X)[:, 1]
        else:
            # For methods that don't support probability
            print(f"Warning: {self.method} does not support probability scores")
            return self.predict(X)
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for scoring.
            
        Returns:
        --------
        np.ndarray
            Anomaly scores.
        """
        if self.method == 'xgboost':
            return self.predict_proba(X)
        elif self.method in ['isolation_forest', 'one_class_svm']:
            # Lower score means more anomalous
            return -self.model.decision_function(X)
        elif self.method == 'lof':
            # Higher score means more anomalous
            return -self.model.decision_function(X)
        elif self.method == 'autoencoder':
            return self.model.decision_function(X)
        else:
            return np.zeros(len(X))
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                param_grid: Dict[str, Any], cv: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for training.
        y : pd.Series
            Target variable.
        param_grid : Dict[str, Any]
            Parameter grid for grid search.
        cv : int, default=5
            Number of cross-validation folds.
            
        Returns:
        --------
        Dict[str, Any]
            Best parameters.
        """
        if self.method != 'xgboost':
            print(f"Warning: Hyperparameter optimization is only supported for XGBoost")
            return {}
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_

def evaluate_model(model: AnomalyDetector, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluate the anomaly detection model.
    
    Parameters:
    -----------
    model : AnomalyDetector
        Trained anomaly detector.
    X : pd.DataFrame
        Features for evaluation.
    y : pd.Series
        True labels.
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with evaluation metrics.
    """
    # Get predictions
    if model.method == 'xgboost':
        y_pred = model.predict(X)
        y_score = model.predict_proba(X)
    else:
        # For unsupervised methods, convert predictions to binary (1 for anomaly, 0 for normal)
        # Note: Isolation Forest and others use -1 for anomalies, 1 for normal
        y_pred = (model.predict(X) == -1).astype(int)
        y_score = model.get_anomaly_scores(X)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(y, y_score)
    except:
        auc = np.nan
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Return metrics
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }