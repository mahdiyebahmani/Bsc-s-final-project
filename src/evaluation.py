from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


CLASS_NAMES = [
    'not_recommended',
    'no_idea',
    'recommended'
]


def evaluate_model(y_test, preds):

    acc = accuracy_score(
        y_test,
        preds
    )

    prec = precision_score(
        y_test,
        preds,
        average='macro'
    )

    rec = recall_score(
        y_test,
        preds,
        average='macro'
    )

    f1 = f1_score(
        y_test,
        preds,
        average='macro'
    )

    report = classification_report(
        y_test,
        preds,
        target_names=CLASS_NAMES
    )

    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Macro F1': f1,
        'Report': report
    }


def print_evaluation(title, y_test, preds):

    metrics = evaluate_model(
        y_test,
        preds
    )

    print(f"\n================ {title} ================")

    print(
        f"Accuracy : {metrics['Accuracy']:.4f}"
    )

    print(
        f"Macro F1 : {metrics['Macro F1']:.4f}"
    )

    print(metrics['Report'])

    return metrics