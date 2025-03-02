import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                           matthews_corrcoef, fowlkes_mallows_score, jaccard_score,
                           log_loss, cohen_kappa_score, precision_recall_curve, roc_curve,
                           brier_score_loss, balanced_accuracy_score)

def decode_confusion_matrix(cm):
    """
    Decodes the confusion matrix and prints detailed information.
    """
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix Details:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

def evaluate_model(ground_truth, predictions):
    """
    Evaluates the model using various metrics and visualizations.
    
    Parameters:
    - ground_truth: List of dictionaries with true labels and label counts.
    - predictions: List of dictionaries with predicted labels and scores.
    """
    # Flatten the ground truth and predictions
    all_true_indices = []
    all_pred_scores = []

    for gt, pred in zip(ground_truth, predictions):
        true_indices = [1 if label in gt['labels'] else 0 for label in pred['label']]
        all_true_indices.extend(true_indices)
        all_pred_scores.extend(pred['scores'])

    # Convert to numpy arrays for convenience
    all_true_indices = np.array(all_true_indices)
    all_pred_scores = np.array(all_pred_scores)
    all_pred_labels = (all_pred_scores >= 0.5).astype(int)

    # Calculate confusion matrix components
    cm = confusion_matrix(all_true_indices, all_pred_labels)
    decode_confusion_matrix(cm)
    
    tn, fp, fn, tp = cm.ravel()

    # Basic metrics
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

    # Advanced metrics
    roc_auc = roc_auc_score(all_true_indices, all_pred_scores)
    pr_auc = average_precision_score(all_true_indices, all_pred_scores)

    mcc = matthews_corrcoef(all_true_indices, all_pred_labels)
    fm = fowlkes_mallows_score(all_true_indices, all_pred_labels)
    jaccard = jaccard_score(all_true_indices, all_pred_labels)

    logloss = log_loss(all_true_indices, all_pred_scores)
    kappa = cohen_kappa_score(all_true_indices, all_pred_labels)
    brier_score = brier_score_loss(all_true_indices, all_pred_scores)

    # Print all metrics
    print(f"\nTrue Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"True Negative Rate (TNR): {tnr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score (Binary): {f1_score_binary:.4f}")
    print(f"Positive Likelihood Ratio (LR+): {lr_plus:.4f}")
    print(f"Negative Likelihood Ratio (LR-): {lr_minus:.4f}")
    print(f"Diagnostic Odds Ratio (DOR): {dor:.4f}")
    print(f"Balanced Accuracy (BACC): {bacc:.4f}")
    print(f"Informedness (INF): {inf:.4f}")
    print(f"Markedness (MK): {mk:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Fowlkes-Mallows Index (FM): {fm:.4f}")
    print(f"Jaccard Index: {jaccard:.4f}")
    print(f"Area Under ROC Curve (AUC-ROC): {roc_auc:.4f}")
    print(f"Area Under Precision-Recall Curve (AUC-PR): {pr_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Brier Score: {brier_score:.4f}")

    # Plotting Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_true_indices, all_pred_scores)
    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(recall, precision, color='b', label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    
    # Plotting ROC Curve
    fpr_roc, tpr_roc, _ = roc_curve(all_true_indices, all_pred_scores)
    plt.subplot(1, 3, 2)
    plt.plot(fpr_roc, tpr_roc, color='r', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='best')

    # Plotting Confusion Matrix
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.show()

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
