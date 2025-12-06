# code/evaluate.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vh_code.features import features
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
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
    feature_names,
):
    
    total_cases = len(y_test)
    actual_hesitant = int((y_test == 1).sum())
    actual_acceptant = total_cases - actual_hesitant

    print("\n==================== CLASSIFICATION EVALUATION ====================")
    print(
        f"\nTest set has {total_cases} cases: "
        f"{actual_hesitant} hesitant (1), {actual_acceptant} not hesitant (0)."
    )

    # Map of model name -> (fitted model, which feature space to use)
    models = {
        "ols": (ols_model, "std"),
        "ridge": (ridge_model, "std"),
        "lasso": (lasso_model, "std"),
        "decision_tree": (dt_model, "raw"),
        "random_forest": (rf_model, "raw"),
    }

    metrics = {}
    predictions = {"y_true": y_test.tolist()}

    # For ROC / PR curves
    roc_data = {}
    pr_data = {}

    # -------------------------------------------------
    # LOOP OVER MODELS: PREDICT + METRICS ON TEST SET
    # -------------------------------------------------
    for name, (model, space) in models.items():
        if space == "std":
            X_eval = X_test_std
        else:
            X_eval = X_test

        # Probabilities for class 1 ("hesitant")
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_eval)[:, 1]
        else:
            scores = model.decision_function(X_eval)
            y_proba = 1.0 / (1.0 + np.exp(-scores))

        # 0.5 threshold
        y_pred = (y_proba >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = float("nan")

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

        metrics[name] = {
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "accuracy": float(acc),
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

        # Store ROC / PR curve data for plotting later
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        roc_data[name] = (fpr_curve, tpr_curve)
        pr_data[name] = (precision_curve, recall_curve)

    # ---------------- SUMMARY TABLE (printed) ----------------
    print("\n---------------- SUMMARY TABLE (AUC / Accuracy) ----------------")
    for name, m in metrics.items():
        print(f"{name:<14s} AUC={m['roc_auc']:.3f}  Acc={m['accuracy']:.3f}")

    # ============================================
    # SAVE METRICS TABLE AS CSV FOR THE REPORT
    # ============================================
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_csv_path = Path("./outputs/metrics_table.csv")
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv_path)
    print(f"\nSaved metrics table to {metrics_csv_path}")

    # ============================================
    # SAVE LOGISTIC REGRESSION COEFFICIENTS
    # ============================================
    coeffs_dir = Path("./outputs")
    coeffs_dir.mkdir(parents=True, exist_ok=True)

    def save_logreg_coeffs(model, model_name, feature_names):
        """
        Saves coefficients of a logistic regression model as CSV.
        Used for interpretability in the report.
        """
        coefs = model.coef_.ravel()
        df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
        })
        df["abs_coefficient"] = df["coefficient"].abs()
        df = df.sort_values("abs_coefficient", ascending=False)
        out_path = coeffs_dir / f"{model_name}_coefficients.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {model_name} coefficients to {out_path}")

    save_logreg_coeffs(ols_model, "ols", feature_names)
    save_logreg_coeffs(ridge_model, "ridge", feature_names)
    save_logreg_coeffs(lasso_model, "lasso", feature_names)

    # ============================================
    # SAVE FEATURE IMPORTANCES (TREE & RF)
    # ============================================
    def save_feature_importances(model, model_name, feature_names):
        """
        Saves tree-based feature importances as CSV.
        Used for interpretability in the report.
        """
        importances = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        df = df.sort_values("importance", ascending=False)
        out_path = coeffs_dir / f"{model_name}_feature_importances.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {model_name} feature importances to {out_path}")

    save_feature_importances(dt_model, "decision_tree", feature_names)
    save_feature_importances(rf_model, "random_forest", feature_names)

    # ============================================
    # SAVE JSON OUTPUTS (metrics + predictions)
    # ============================================
    metrics_path = Path("./outputs/metrics.json")
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    preds_path = Path("./outputs/predictions.json")
    with preds_path.open("w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {preds_path}")

    # ============================================
    # FEATURE IMPORTANCE + ROC/PR
    # ============================================
    figures_dir = Path("./outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Feature importance bar chart (Random Forest, top 15) ----
    rf_importances = rf_model.feature_importances_
    rf_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_importances,
    }).sort_values("importance", ascending=False)

    top_k = 15
    rf_top = rf_df.head(top_k)

    plt.figure(figsize=(8, 6))
    plt.barh(rf_top["feature"], rf_top["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance (top 15)")
    plt.tight_layout()
    rf_fig_path = figures_dir / "random_forest_feature_importance_top15.png"
    plt.savefig(rf_fig_path, dpi=300)
    plt.close()
    print(f"Saved Random Forest feature importance plot to {rf_fig_path}")

    # ---- ROC curves for all models ----
    plt.figure(figsize=(8, 6))
    for name, (fpr_curve, tpr_curve) in roc_data.items():
        plt.plot(fpr_curve, tpr_curve, label=f"{name} (AUC={metrics[name]['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves (test set)")
    plt.legend()
    plt.tight_layout()
    roc_fig_path = figures_dir / "roc_curves.png"
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves plot to {roc_fig_path}")

    # ---- Precision–Recall curves for all models ----
    plt.figure(figsize=(8, 6))
    for name, (precision_curve, recall_curve) in pr_data.items():
        plt.plot(recall_curve, precision_curve, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curves (test set)")
    plt.legend()
    plt.tight_layout()
    pr_fig_path = figures_dir / "precision_recall_curves.png"
    plt.savefig(pr_fig_path, dpi=300)
    plt.close()
    print(f"Saved Precision–Recall curves plot to {pr_fig_path}\n")
    
    
def wave1_prediction_analysis(
    ols_model,
    ridge_model,
    lasso_model,
    dt_model,
    rf_model,
    X_train_raw,
    y_train_raw,
    X_train_fe,
    X_train_std_fe,
    vh_wave1,
):
    

    X_wave1_raw = vh_wave1.copy()
    y_dummy = np.zeros(len(X_wave1_raw))  # dummy target to reuse features()


    for col in X_train_raw.columns:
        if col not in X_wave1_raw.columns:
            X_wave1_raw[col] = np.nan

    (
        X_train_fe_w1,
        X_wave1_fe,
        _,
        _,
        X_train_std_w1,
        X_wave1_std_w1,
    ) = features(
        X_train_raw.copy(),
        X_wave1_raw.copy(),
        y_train_raw.copy(),
        y_dummy,
    )

    X_wave1_feat_aligned = X_wave1_fe.reindex(columns=X_train_fe.columns, fill_value=0.0)

    X_wave1_std_aligned = X_wave1_std_w1.reindex(columns=X_train_std_fe.columns, fill_value=0.0)

    wave1_predictions = {
        "ols": ols_model.predict_proba(X_wave1_std_aligned)[:, 1],
        "ridge": ridge_model.predict_proba(X_wave1_std_aligned)[:, 1],
        "lasso": lasso_model.predict_proba(X_wave1_std_aligned)[:, 1],
        "decision_tree": dt_model.predict_proba(X_wave1_feat_aligned)[:, 1],
        "random_forest": rf_model.predict_proba(X_wave1_feat_aligned)[:, 1],
    }

    wave1_df = pd.DataFrame(wave1_predictions)
    wave1_df["respondent_id"] = vh_wave1.index

    outputs_dir = Path("./outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    wave1_path = outputs_dir / "wave1_predictions.csv"
    wave1_df.to_csv(wave1_path, index=False)
    print(f"Saved Wave 1 predictions to {wave1_path}")

    # ----------------------------
    # BASIC DESCRIPTIVE STAT
    # ----------------------------
    print("\n===== DESCRIPTIVE STATS: WAVE 1 PREDICTIONS =====")
    desc_stats = wave1_df.drop(columns=["respondent_id"]).describe()
    desc_path = outputs_dir / "wave1_prediction_descriptives.csv"
    desc_stats.to_csv(desc_path)
    print(desc_stats)
    print(f"\nSaved Wave 1 descriptive statistics to {desc_path}")

    # ----------------------------
    # DISTRIBUTION PLOTS (HISTOS)
    # ----------------------------
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for col in ["ols", "ridge", "lasso", "decision_tree", "random_forest"]:
        plt.figure(figsize=(7, 5))
        plt.hist(wave1_df[col], bins=30)
        plt.xlabel("Predicted probability of hesitancy")
        plt.ylabel("Frequency")
        plt.title(f"Wave 1 prediction distribution — {col}")
        plt.tight_layout()
        fig_path = figures_dir / f"wave1_distribution_{col}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved Wave 1 distribution plot for {col} to {fig_path}")

    print("\n===== WAVE 1 PREDICTION ANALYSIS DONE =====\n")