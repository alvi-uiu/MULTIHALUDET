import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, roc_curve, 
                             average_precision_score, confusion_matrix, brier_score_loss, 
                             log_loss, matthews_corrcoef, cohen_kappa_score, 
                             balanced_accuracy_score, fbeta_score)

def compute_uncertainty_metrics(probs, labels, preds):
    predictive_entropy = -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(1 - probs + 1e-10)
    confidence = np.maximum(probs, 1 - probs)
    uncertainty = 1 - confidence
    
    brier = brier_score_loss(labels, probs)
    logloss = log_loss(labels, probs)
    
    correct = (preds == labels)
    ece = 0
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidence[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
            
    return {
        'predictive_entropy': predictive_entropy,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'brier_score': brier,
        'log_loss': logloss,
        'ece': ece
    }

def find_best_thresholds(probs, labels):
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    youden_j = tpr - fpr
    best_threshold_youden = thresholds_roc[np.argmax(youden_j)]
    
    best_f1 = 0
    best_threshold_f1 = 0.5
    
    for thresh in np.linspace(0.1, 0.9, 81):
        preds_temp = (probs >= thresh).astype(int)
        f1_temp = f1_score(labels, preds_temp)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold_f1 = thresh
            
    return {
        'youden': best_threshold_youden,
        'f1': best_threshold_f1
    }

def evaluate_all(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    ap = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    
    cm = confusion_matrix(labels, preds)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-10)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-10)
    
    return {
        'auc': auc, 'f1': f1, 'acc': acc, 'ap': ap, 'mcc': mcc, 
        'kappa': kappa, 'bal_acc': bal_acc, 'sensitivity': sensitivity, 
        'specificity': specificity, 'cm': cm
    }
