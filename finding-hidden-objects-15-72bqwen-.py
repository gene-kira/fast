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
        'TPR': tpr,
        'FPR': fpr,
        'TNR': tnr,
        'FNR': fnr,
        'Precision': precision,
        'Recall': recall,
        'F1 Score (Binary)': f1_score_binary,
        'Positive Likelihood Ratio (LR+)': lr_plus,
        'Negative Likelihood Ratio (LR-)': lr_minus,
        'Diagnostic Odds Ratio (DOR)': dor,
        'Balanced Accuracy (BACC)': bacc,
        'Informedness (INF)': inf,
        'Markedness (MK)': mk
    }

def calculate_advanced_metrics(all_true_indices, all_pred_scores, all_pred_labels):
    """
    Calculate advanced metrics.
    
    Parameters:
    - all_true_indices: Ground truth labels (1D array)
    - all_pred_scores: Prediction scores (1D array)
    - all_pred_labels: Predicted labels (1D array)
    
    Returns:
    - Dictionary of advanced metrics
    """
    roc_auc = roc_auc_score(all_true_indices, all_pred_scores)
    pr_auc = average_precision_score(all_true_indices, all_pred_scores)

    mcc = matthews_corrcoef(all_true_indices, all_pred_labels)
    fm = fowlkes_mallows_score(all_true_indices, all_pred_labels)
    jaccard = jaccard_score(all_true_indices, all_pred_labels)

    logloss = log_loss(all_true_indices, all_pred_scores)
    kappa = cohen_kappa_score(all_true_indices, all_pred_labels)
    brier_score = brier_score_loss(all_true_indices, all_pred_scores)

    return {
        'AUC-ROC': roc_auc,
        'AUC-PR': pr_auc,
        'Matthews Correlation Coefficient (MCC)': mcc,
        'Fowlkes-Mallows Index (FM)': fm,
        'Jaccard Index': jaccard,
        'Log Loss': logloss,
        'Cohen\'s Kappa': kappa,
        'Brier Score': brier_score
    }

def plot_metrics(cm, all_true_indices, all_pred_scores):
    """
    Plot confusion matrix and curves.
    
    Parameters:
    - cm: Confusion matrix (2x2 array)
    - all_true_indices: Ground truth labels (1D array)
    - all_pred_scores: Prediction scores (1D array)
    """
    precision, recall, _ = precision_recall_curve(all_true_indices, all_pred_scores)
    fpr_roc, tpr_roc, _ = roc_curve(all_true_indices, all_pred_scores)

    plt.figure(figsize=(20, 6))

    # Precision-Recall Curve
    plt.subplot(1, 3, 1)
    plt.plot(recall, precision, color='b', label=f'PR curve (area = {average_precision_score(all_true_indices, all_pred_scores):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')

    # ROC Curve
    plt.subplot(1, 3, 2)
    plt.plot(fpr_roc, tpr_roc, color='r', label=f'ROC curve (area = {roc_auc_score(all_true_indices, all_pred_scores):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='best')

    # Confusion Matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

def evaluate_model(ground_truth, predictions):
    """
    Evaluates the model using various metrics and visualizations.
    
    Parameters:
    - ground_truth: List of dictionaries with true labels and label counts
    - predictions: List of dictionaries with predicted labels and scores
    """
    try:
        # Validate input lengths
        if len(ground_truth) != len(predictions):
            raise ValueError("Lengths of ground truth and predictions do not match.")

        all_true_indices = []
        all_pred_scores = []

        for gt, pred in zip(ground_truth, predictions):
            labels_gt = gt['labels']
            labels_pred = pred['label']

            # Flatten the true labels
            all_true_indices.extend([1 if label in labels_gt else 0 for label in labels_pred])

            # Use the scores directly
            all_pred_scores.extend(pred['scores'])

        all_pred_labels = [1 if score >= 0.5 else 0 for score in all_pred_scores]

        cm = confusion_matrix(all_true_indices, all_pred_labels)
        decode_confusion_matrix(cm)

        basic_metrics = calculate_basic_metrics(cm)
        advanced_metrics = calculate_advanced_metrics(all_true_indices, all_pred_scores, all_pred_labels)

        logging.info("Basic Metrics:")
        for key, value in basic_metrics.items():
            logging.info(f"{key}: {value}")

        logging.info("Advanced Metrics:")
        for key, value in advanced_metrics.items():
            logging.info(f"{key}: {value}")

        plot_metrics(cm, all_true_indices, all_pred_scores)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Example ground truth and predictions
ground_truth = [
    {'labels': [1, 2], 'label_count': 2},
    {'labels': [3], 'label_count': 1}
]

predictions = [
    {'label': [1, 2], 'scores': [0.8, 0.7]},
    {'label': [3], 'scores': [0.9]}
]

# Evaluate the model
evaluate_model(ground_truth, predictions)
