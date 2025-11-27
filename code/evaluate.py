# code/evaluate.py

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate(
    ols_model,
    ridge_model,
    lasso_model,
    dt_model,
    rf_model,
    X_test,
    X_test_std,
    y_test,
):
    total_cases = len(y_test)
    actual_hesitant = int((y_test == 1).sum())
    actual_acceptant = total_cases - actual_hesitant

    print("\n==================== CLASSIFICATION EVALUATION ====================")
    print(
        f"\nTest set has {total_cases} cases: "
        f"{actual_hesitant} hesitant (1), {actual_acceptant} not hesitant (0)."
    )

    models = {
        "ols": (ols_model, "std"),
        "ridge": (ridge_model, "std"),
        "lasso": (lasso_model, "std"),
        "decision_tree": (dt_model, "raw"),
        "random_forest": (rf_model, "raw"),
    }

    metrics = {}
    predictions = {"y_true": y_test.tolist()}

    for name, (model, space) in models.items():
        if space == "std":
            X_eval = X_test_std
        else:
            X_eval = X_test

        # Predicted probabilities for class 1 ("hesitant")
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_eval)[:, 1]
        else:
            scores = model.decision_function(X_eval)
            y_proba = 1.0 / (1.0 + np.exp(-scores))

        # 0.5 threshold to classify hesitant vs not
        y_pred = (y_proba >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)  # sensitivity for class 1
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC: requires both classes present; handle edge case
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = float("nan")

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

        predicted_hesitant = int((y_pred == 1).sum())
        predicted_acceptant = int((y_pred == 0).sum())

        metrics[name] = {
            "predicted_hesitant": int(predicted_hesitant),
            "predicted_not_hesitant": int(predicted_acceptant),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "accuracy": float(acc),
            "accuracy_percent": float(acc * 100.0),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(auc),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "total_cases": int(total_cases),
            "actual_hesitant": int(actual_hesitant),
            "actual_not_hesitant": int(actual_acceptant),
        }

        predictions[name] = {
            "y_proba": y_proba.tolist(),
            "y_pred": y_pred.tolist(),
        }

    print("\n---------------- SUMMARY TABLE ----------------")

    header_cols = [
        "Model",
        "Pred_Hes",
        "Pred_NotHes",
        "TP",
        "FN",
        "FP",
        "TN",
        "Acc",
        "Prec",
        "Rec",
        "F1",
        "AUC",
        "FPR",
        "FNR",
    ]

    header_line = (
        f"{header_cols[0]:<14s}"
        f"{header_cols[1]:>10s}"
        f"{header_cols[2]:>13s}"
        f"{header_cols[3]:>6s}"
        f"{header_cols[4]:>6s}"
        f"{header_cols[5]:>6s}"
        f"{header_cols[6]:>6s}"
        f"{header_cols[7]:>8s}"
        f"{header_cols[8]:>8s}"
        f"{header_cols[9]:>8s}"
        f"{header_cols[10]:>8s}"
        f"{header_cols[11]:>8s}"
        f"{header_cols[12]:>8s}"
        f"{header_cols[13]:>8s}"
    )
    print(header_line)
    print("-" * len(header_line))

    for name, m in metrics.items():
        row = (
            f"{name:<14s}"
            f"{m['predicted_hesitant']:>10d}"
            f"{m['predicted_not_hesitant']:>13d}"
            f"{m['tp']:>6d}"
            f"{m['fn']:>6d}"
            f"{m['fp']:>6d}"
            f"{m['tn']:>6d}"
            f"{m['accuracy']*100:>7.2f}%"
            f"{m['precision']:>8.4f}"
            f"{m['recall']:>8.4f}"
            f"{m['f1']:>8.4f}"
            f"{m['roc_auc']:>8.4f}"
            f"{m['fpr']:>8.4f}"
            f"{m['fnr']:>8.4f}"
        )
        print(row)

    # ---------- Save metrics.json ----------
    metrics_path = Path("./outputs/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # ---------- Save predictions.json ----------
    preds_path = Path("./outputs/predictions.json")
    with preds_path.open("w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {preds_path}\n")