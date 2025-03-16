import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    confusion_matrix, classification_report
)

def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, 
                               title: str = 'Precision-Recall Curve') -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels.
    y_score : np.ndarray
        Predicted scores.
    title : str, default='Precision-Recall Curve'
        Title for the plot.
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    return fig

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, 
                  title: str = 'ROC Curve') -> plt.Figure:
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels.
    y_score : np.ndarray
        Predicted scores.
    title : str, default='ROC Curve'
        Title for the plot.
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = 'Confusion Matrix') -> plt.Figure:
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    title : str, default='Confusion Matrix'
        Title for the plot.
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    
    return fig

def plot_feature_importance(model, feature_names: List[str], 
                           title: str = 'Feature Importance') -> plt.Figure:
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute.
    feature_names : List[str]
        Names of features.
    title : str, default='Feature Importance'
        Title for the plot.
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure.
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    top_n = min(20, len(feature_names))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(top_n), importances[indices][:top_n], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n]])
    ax.set_xlabel('Feature Importance')
    ax.set_title(title)
    
    return fig

def compare_models(models: Dict[str, Dict[str, float]], 
                  metrics: List[str] = ['precision', 'recall', 'f1_score', 'auc']) -> plt.Figure:
    """
    Compare multiple models based on evaluation metrics.
    
    Parameters:
    -----------
    models : Dict[str, Dict[str, float]]
        Dictionary with model names as keys and evaluation metrics as values.
    metrics : List[str], default=['precision', 'recall', 'f1_score', 'auc']
        List of metrics to compare.
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure.
    """
    # Prepare data for plotting
    data = []
    for model_name, model_metrics in models.items():
        for metric in metrics:
            if metric in model_metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': model_metrics[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df, ax=ax)
    ax.set_title('Model Comparison')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y')
    
    return fig

def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray, 
                          metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels.
    y_score : np.ndarray
        Predicted scores.
    metric : str, default='f1'
        Metric to optimize ('f1', 'precision', 'recall').
        
    Returns:
    --------
    Tuple[float, float]
        Optimal threshold and corresponding metric value.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # Add threshold=1.0 to match the length of precision and recall
    thresholds = np.append(thresholds, 1.0)
    
    if metric == 'f1':
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx], f1_scores[optimal_idx]
    
    elif metric == 'precision':
        optimal_idx = np.argmax(precision)
        return thresholds[optimal_idx], precision[optimal_idx]
    
    elif metric == 'recall':
        optimal_idx = np.argmax(recall)
        return thresholds[optimal_idx], recall[optimal_idx]
    
    else:
        raise ValueError(f"Unknown metric: {metric}")