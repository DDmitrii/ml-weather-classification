from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def compute_metrics(all_labels, all_preds, average='macro'):
    """Compute classification metrics"""
    metrics = {
        'precision': precision_score(all_labels, all_preds, average=average, zero_division=0),
        'recall': recall_score(all_labels, all_preds, average=average, zero_division=0),
        'f1': f1_score(all_labels, all_preds, average=average, zero_division=0)
    }
    return metrics


def print_per_class_metrics(all_labels, all_preds, class_names):
    """Print per-class metrics"""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<8}")