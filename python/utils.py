import torch

from sklearn.metrics import (PrecisionRecallDisplay, accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)

import numpy as np


def calc_f1(p_and_r):
    p, r = p_and_r
    return (2*p*r)/(p+r)

def print_model_metrics_ODF(y_test, y_test_prob, matching_distance = 1, verbose = True, return_metrics = True, plot = False):

    if matching_distance > 1:
        kernel = torch.ones(1, matching_distance, matching_distance)
        y_test = torch.nn.functional.conv2d(y_test.float(), kernel.unsqueeze(1), padding='same')
        y_test = torch.clip(y_test, 0, 1)

    y_test = y_test.flatten()
    y_test_prob = y_test_prob.flatten()

    precision, recall, threshold = precision_recall_curve(y_test, y_test_prob, pos_label = 1)
    
    #Find the threshold value that gives the best F1 Score
    best_f1_index = np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])
    best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]
    
    # Calulcate predictions based on the threshold value
    y_test_pred = np.where(y_test_prob > best_threshold, 1, 0)
    
    # Calculate all metrics
    f1 = f1_score(y_test, y_test_pred, pos_label = 1, average = 'binary')
    roc_auc = roc_auc_score(y_test, y_test_prob)
    acc = accuracy_score(y_test, y_test_pred)
        
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} | Best threshold: {:.3f} \
            n'.format(f1, best_precision, best_recall, roc_auc, acc, best_threshold))

    if plot:
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    
    if return_metrics:
        return np.array([f1, best_precision, best_recall, roc_auc, acc, best_threshold])

        
def print_model_metrics_OIF(y_tests, y_test_probs, matching_distance = 1, verbose = True, return_metrics = True):

    f1s = []
    best_precisions = []
    best_recalls = []
    # roc_aucs = []
    accs = []
    best_thresholds = []

    for y_test, y_test_prob in zip(y_tests, y_test_probs):
        if matching_distance > 1:
            kernel = torch.ones(1, matching_distance, matching_distance)
            y_test = torch.nn.functional.conv2d(y_test.float().unsqueeze(1), kernel.unsqueeze(1), padding='same')
            y_test = torch.clip(y_test, 0, 1).squeeze(1)

        y_test = y_test.flatten()
        y_test_prob = y_test_prob.flatten()

        precision, recall, threshold = precision_recall_curve(y_test, y_test_prob, pos_label = 1)
        
        #Find the threshold value that gives the best F1 Score
        best_f1_index = np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])
        best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]
        
        # Calulcate predictions based on the threshold value
        y_test_pred = np.where(y_test_prob > best_threshold, 1, 0)
        
        # Calculate all metrics
        f1 = f1_score(y_test, y_test_pred, pos_label = 1, average = 'binary', zero_division=0)
        # roc_auc = roc_auc_score(y_test, y_test_prob)
        acc = accuracy_score(y_test, y_test_pred)
        
        f1s += [f1]
        best_precisions += [best_precision]
        best_recalls += [best_recall]
        # roc_aucs += [roc_auc]
        accs += [acc]
        best_thresholds += [best_threshold]
        
    f1 = np.mean(f1)
    best_precision = np.mean(best_precision)
    best_recall = np.mean(best_recall)
    # roc_auc = np.mean(roc_auc)
    acc = np.mean(acc)
    best_threshold = np.mean(best_threshold)

    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f}  | Accuracy: {:.3f} | Best threshold: {:.3f} \
            n'.format(f1, best_precision, best_recall, acc, best_threshold))
    
    if return_metrics:
        return np.array([f1, best_precision, best_recall, acc, best_threshold])