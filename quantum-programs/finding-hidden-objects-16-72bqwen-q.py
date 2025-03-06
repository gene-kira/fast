import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             matthews_corrcoef, fowlkes_mallows_score, jaccard_score,
                             log_loss, cohen_kappa_score, precision_recall_curve, roc_curve,
                             brier_score_loss, balanced_accuracy_score)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def decode_confusion_matrix(cm):
    """
    Decodes the confusion matrix and prints detailed information.
    
    Parameters:
    - cm: Confusion matrix (2x2 array)
    """
    tn, fp, fn, tp = cm.ravel()
    
    logging.info("Confusion Matrix Details:")
    logging.info(f"True Positives (TP): {tp}")
    logging.info(f"False Positives (FP): {fp}")
    logging.info(f"True Negatives (TN): {tn}")
    logging.info(f"False Negatives (FN): {fn}")

def calculate_basic_metrics(cm):
    """
    Calculate basic metrics from the confusion matrix.
    
    Parameters:
    - cm: Confusion matrix (2x2 array)
    
    Returns:
    - Dictionary of basic metrics
    """
    tn, fp, fn, tp = cm.ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1_score_binary = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    lr_plus = tpr / fpr if fpr > 0 else float('inf')
    lr_minus = fnr / tnr if tnr > 0 else float('inf')
    dor = lr_plus / lr_minus if lr_minus != 0 else float('inf')

    bacc = balanced_accuracy_score(all_true_indices, all_pred_labels)
    inf = tpr + tnr - 1
    mk = precision + (tn / (tn + fn)) - 1
    
    return {
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn,
        'True Positive Rate': tpr,
        'False Positive Rate': fpr,
        'True Negative Rate': tnr,
        'False Negative Rate': fnr,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score_binary,
        'Positive Likelihood Ratio': lr_plus,
        'Negative Likelihood Ratio': lr_minus,
        'Diagnostic Odds Ratio': dor,
        'Balanced Accuracy': bacc,
        'Informedness': inf,
        'Markedness': mk
    }

def calculate_advanced_metrics(all_true_indices, all_pred_scores, all_pred_labels):
    """
    Calculate advanced metrics from true and predicted values.
    
    Parameters:
    - all_true_indices: True labels (1D array)
    - all_pred_scores: Predicted scores (1D array)
    - all_pred_labels: Predicted labels (1D array)
    
    Returns:
    - Dictionary of advanced metrics
    """
    cm = confusion_matrix(all_true_indices, all_pred_labels)
    basic_metrics = calculate_basic_metrics(cm)
    
    roc_auc = roc_auc_score(all_true_indices, all_pred_scores)
    pr_auc = average_precision_score(all_true_indices, all_pred_scores)
    mcc = matthews_corrcoef(all_true_indices, all_pred_labels)
    f1_micro = f1_score(all_true_indices, all_pred_labels, average='micro')
    f1_macro = f1_score(all_true_indices, all_pred_labels, average='macro')
    
    return {
        **basic_metrics,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'Matthews Correlation Coefficient': mcc,
        'F1 Score (Micro)': f1_micro,
        'F1 Score (Macro)': f1_macro
    }

def plot_metrics(cm, all_true_indices, all_pred_scores):
    """
    Plot confusion matrix and ROC curve.
    
    Parameters:
    - cm: Confusion matrix (2D array)
    - all_true_indices: True labels (1D array)
    - all_pred_scores: Predicted scores (1D array)
    """
    plt.figure(figsize=(10, 5))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_true_indices, all_pred_scores)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(all_true_indices, all_pred_scores)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def combine_predictions(predictions_list):
    """
    Combine predictions from multiple models using a weighted average.
    
    Parameters:
    - predictions_list: List of dictionaries with predicted labels and scores
    
    Returns:
    - Combined predictions (1D array)
    """
    all_pred_scores = []
    for pred in predictions_list:
        all_pred_scores.append(pred['scores'])
    
    # Combine scores using a simple mean
    combined_scores = np.mean(all_pred_scores, axis=0)
    
    return combined_scores

def apply_attention(visual_features, audio_features):
    """
    Apply an attention mechanism to combine visual and audio features.
    
    Parameters:
    - visual_features: Visual features (2D array)
    - audio_features: Audio features (2D array)
    
    Returns:
    - Combined features with attention (2D array)
    """
    # Example attention mechanism: simple weighted sum
    combined_features = 0.6 * visual_features + 0.4 * audio_features
    
    return combined_features

def detect_temporal_anomalies(features, window_size=5, threshold=2):
    """
    Detect temporal anomalies in features.
    
    Parameters:
    - features: Features to analyze (1D or 2D array)
    - window_size: Size of the sliding window for anomaly detection
    - threshold: Z-score threshold for detecting anomalies
    
    Returns:
    - Anomaly indices
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = np.convolve(features, np.ones(window_size), mode='valid') / window_size
    rolling_std = np.sqrt(np.convolve((features - rolling_mean) ** 2, np.ones(window_size), mode='valid') / window_size)
    
    # Calculate z-scores
    z_scores = (features[window_size-1:] - rolling_mean) / rolling_std
    
    # Identify anomalies
    anomaly_indices = np.where(np.abs(z_scores) > threshold)[0] + window_size - 1
    
    return anomaly_indices

def evaluate_model(ground_truth, predictions_list, visual_features=None, audio_features=None):
    """
    Evaluate the model using quantum-inspired techniques and temporal anomaly detection.
    
    Parameters:
    - ground_truth: List of dictionaries with true labels and label counts
    - predictions_list: List of dictionaries with predicted labels and scores
    - visual_features: Visual features (2D array, optional)
    - audio_features: Audio features (2D array, optional)
    """
    try:
        # Validate input lengths
        if len(ground_truth) != len(predictions_list):
            raise ValueError("Lengths of ground truth and predictions do not match.")

        all_true_indices = []
        all_pred_scores = []

        for gt, pred in zip(ground_truth, predictions_list):
            labels_gt = gt['labels']
            labels_pred = pred['label']

            # Flatten the true labels
            all_true_indices.extend([1 if label in labels_gt else 0 for label in labels_pred])

            # Use the scores directly
            all_pred_scores.extend(pred['scores'])

        all_pred_labels = [1 if score >= 0.5 else 0 for score in all_pred_scores]

        cm = confusion_matrix(all_true_indices, all_pred_labels)
        decode_confusion_matrix(cm)

        combined_scores = combine_predictions(predictions_list)
        all_pred_labels_combined = [1 if score >= 0.5 else 0 for score in combined_scores]

        basic_metrics = calculate_basic_metrics(cm)
        advanced_metrics = calculate_advanced_metrics(all_true_indices, combined_scores, all_pred_labels_combined)

        logging.info("Basic Metrics:")
        for key, value in basic_metrics.items():
            logging.info(f"{key}: {value}")

        logging.info("Advanced Metrics:")
        for key, value in advanced_metrics.items():
            logging.info(f"{key}: {value}")

        plot_metrics(cm, all_true_indices, combined_scores)

        if visual_features is not None and audio_features is not None:
            combined_features = apply_attention(visual_features, audio_features)
            anomaly_indices = detect_temporal_anomalies(combined_features)
            logging.info(f"Temporal Anomaly Indices: {anomaly_indices}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Example ground truth and predictions
ground_truth = [
    {'labels': [1, 2], 'label_count': 2},
    {'labels': [3], 'label_count': 1}
]

predictions_list = [
    {'label': [1, 2], 'scores': [0.8, 0.7]},
    {'label': [3], 'scores': [0.9]}
]

# Example visual and audio features
visual_features = np.random.rand(10, 5)  # 10 frames with 5 visual features each
audio_features = np.random.rand(10, 5)   # 10 frames with 5 audio features each

# Evaluate the model
evaluate_model(ground_truth, predictions_list, visual_features, audio_features)
