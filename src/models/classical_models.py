import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def get_classical_models():

    models = [
        (
            'Naive Bayes',
            MultinomialNB(alpha=0.1)
        ),

        (
            'Logistic Regression',
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced'
            )
        ),

        (
            'Linear SVM',
            LinearSVC(
                C=1.0,
                max_iter=5000,
                class_weight='balanced'
            )
        )
    ]

    return models


def run_classical_models(
    feature_sets,
    y_train,
    y_test,
    evaluate_function
):

    results = []

    models = get_classical_models()

    for model_name, model in models:

        for (
            vec_type,
            feat_name
        ), (
            X_tr,
            X_te
        ) in feature_sets.items():

            model.fit(X_tr, y_train)

            preds = model.predict(X_te)

            metrics = evaluate_function(
                y_test,
                preds
            )

            results.append({
                'Model': model_name,
                'Feature': (
                    f"{vec_type.upper()} "
                    f"{feat_name}"
                ),
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'Macro F1': metrics['Macro F1'],
                'Report': metrics['Report']
            })

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            'Macro F1',
            ascending=False
        )
        .reset_index(drop=True)
    )

    return results_df