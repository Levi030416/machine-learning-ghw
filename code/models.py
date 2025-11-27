# code/models.py

import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import optuna


def _tune_ridge_classifier(seed, TRIALS_NUMBER, K_FOLD_K, X_train_std, y_train):

    start = time.perf_counter()

    cv = KFold(n_splits=K_FOLD_K, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        # Sample inverse regularization parameter C
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)

        model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        # Cross-validated ROC-AUC
        scores = cross_val_score(
            model,
            X_train_std,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return scores.mean()

    # Optuna with fixed seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=TRIALS_NUMBER, show_progress_bar=False)

    best_C = study.best_params["C"]
    best_lambda = 1.0 / best_C

    # Fit final model
    best_model = LogisticRegression(
        penalty="l2",
        C=best_C,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    best_model.fit(X_train_std, y_train)

    end = time.perf_counter()
    print(f"[Ridge]        Completed in {end - start:.2f} s. Best lambda: {best_lambda:.6f}. Starting Lasso...")

    return best_model


def _tune_lasso_classifier(seed, TRIALS_NUMBER, K_FOLD_K, X_train_std, y_train):

    start = time.perf_counter()

    cv = KFold(n_splits=K_FOLD_K, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)

        model = LogisticRegression(
            penalty="l1",
            C=C,
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )

        scores = cross_val_score(
            model,
            X_train_std,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=TRIALS_NUMBER, show_progress_bar=False)

    best_C = study.best_params["C"]
    best_lambda = 1.0 / best_C

    best_model = LogisticRegression(
        penalty="l1",
        C=best_C,
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    best_model.fit(X_train_std, y_train)

    end = time.perf_counter()
    print(f"[Lasso]        Completed in {end - start:.2f} s. Best lambda: {best_lambda:.6f}. Starting Decision Tree...")

    return best_model


def _tune_decision_tree_classifier(seed, TRIALS_NUMBER, K_FOLD_K, X_train, y_train):

    start = time.perf_counter()

    cv = KFold(n_splits=K_FOLD_K, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=TRIALS_NUMBER, show_progress_bar=False)

    best_params = study.best_params

    best_model = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=seed,
    )
    best_model.fit(X_train, y_train)

    end = time.perf_counter()
    print(f"[DecisionTree] Completed in {end - start:.2f} s. Best params: {best_params}. Starting Random Forest...")

    return best_model


def _tune_random_forest_classifier(seed, TRIALS_NUMBER, K_FOLD_K, X_train, y_train):

    start = time.perf_counter()

    cv = KFold(n_splits=K_FOLD_K, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=seed,
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=TRIALS_NUMBER, show_progress_bar=False)

    best_params = study.best_params

    best_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        n_jobs=-1,
        random_state=seed,
    )
    best_model.fit(X_train, y_train)

    end = time.perf_counter()
    print(f"[RandomForest] Completed in {end - start:.2f} s. Best params: {best_params}. No more models to run.")

    return best_model


def models(seed, TRIALS_NUMBER, K_FOLD_K, X_train, y_train, X_train_std):

    print("Beginning model training...\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # OLS baseline
    start_ols = time.perf_counter()
    ols_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    ols_model.fit(X_train_std, y_train)
    end_ols = time.perf_counter()
    print(f"[OLS]          Completed in {end_ols - start_ols:.2f} s. Starting Ridge...")

    # Sequential tuning of ML models
    ridge_model = _tune_ridge_classifier(seed, TRIALS_NUMBER, K_FOLD_K, X_train_std, y_train)
    lasso_model = _tune_lasso_classifier(seed, TRIALS_NUMBER, K_FOLD_K, X_train_std, y_train)
    dt_model = _tune_decision_tree_classifier(seed, TRIALS_NUMBER, K_FOLD_K, X_train, y_train)
    rf_model = _tune_random_forest_classifier(seed, TRIALS_NUMBER, K_FOLD_K, X_train, y_train)

    return ols_model, ridge_model, lasso_model, dt_model, rf_model
