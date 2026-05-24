import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, calibration_curve

def plot_publication_results(probs, labels, preds, uncertainty_metrics, auc_score, ap_score, cm, filename):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(2, 3, 1)
    fpr, tpr, _ = roc_curve(labels, probs)
    ax1.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC (AUC = {auc_score:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('(A) ROC Curve')
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)
    precision, recall, _ = precision_recall_curve(labels, probs)
    ax2.plot(recall, precision, 'g-', linewidth=2.5, label=f'PR (AP = {ap_score:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('(B) Precision-Recall Curve')
    ax2.legend()

    ax3 = fig.add_subplot(2, 3, 3)
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    ax3.plot(prob_pred, prob_true, 's-', label='Model')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Fraction of Positives')
    ax3.set_title(f'(C) Calibration (ECE={uncertainty_metrics["ece"]:.3f})')
    ax3.legend()

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(cm, interpolation='nearest', cmap='Blues')
    ax4.set_title('(D) Confusion Matrix')
    classes = ['Non-Hall.', 'Hall.']
    tick_marks = np.arange(len(classes))
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(classes)
    ax4.set_yticklabels(classes)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(probs[labels == 0], bins=30, alpha=0.6, label='Non-Hall.', density=True)
    ax5.hist(probs[labels == 1], bins=30, alpha=0.6, label='Hall.', density=True)
    ax5.set_title('(E) Distribution')
    ax5.legend()

    ax6 = fig.add_subplot(2, 3, 6)
    correct_mask = (preds == labels)
    ax6.hist(uncertainty_metrics['confidence'][correct_mask], bins=25, alpha=0.6, label='Correct', density=True)
    ax6.hist(uncertainty_metrics['confidence'][~correct_mask], bins=25, alpha=0.6, label='Incorrect', density=True)
    ax6.set_title('(F) Confidence')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
