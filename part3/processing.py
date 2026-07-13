"""
processing.py

Part 2 pipeline: supervised ML on top of a Part 1 cleaned_data.csv.

Builds two models on a shared, leak-free feature matrix:
  - Regression:     word_count (continuous)              via Linear + Ridge
  - Classification: Spam/Ham binarized (spam=1/ham=0)     via Logistic Regression

Reuses the run_dir / tmp-output pattern and logging conventions from processing.py
so charts and artifacts for a Part 2 run land next to the Part 1 run they came from.
"""
import io
import os
import base64
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score,
)

from logger_setup import get_logger

logger = get_logger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "tmp"))
RUN_DIR_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "run"))
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(RUN_DIR_ROOT, exist_ok=True)

BASE_NUMERIC_FEATURES = ["avg_word_len", "capital_ratio", "digit_count", "punct_count", "exclaim_count"]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
IMBALANCE_THRESHOLD_PCT = 35.0

NUMERIC_COLS = [
    "word_count", "char_count", "avg_word_len",
    "capital_ratio", "digit_count", "punct_count", "exclaim_count",
]

REQUIRED_COLS = {"Subject", "Message", "Spam/Ham"}


class PipelineError(ValueError):
    """Raised for user-facing, expected problems with the uploaded file."""


def load_and_clean(file_like):
    logger.info("ACTION load_and_clean: parsing uploaded CSV")
    try:
        df = pd.read_csv(file_like)
    except Exception as exc:
        logger.error("load_and_clean: failed to parse CSV (%s)", exc)
        raise PipelineError(f"Could not parse file as CSV ({exc}).")

    logger.info("load_and_clean: parsed CSV with shape=%s columns=%s", df.shape, list(df.columns))

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        logger.error("load_and_clean: missing required column(s): %s", sorted(missing))
        raise PipelineError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}. "
            f"Expected at least: {', '.join(sorted(REQUIRED_COLS))}."
        )

    rows_before = len(df)
    df = df.drop_duplicates(subset=["Subject", "Message", "Spam/Ham"], keep="first")
    logger.info("load_and_clean: dropped %d content-duplicate rows (%d -> %d)",
                rows_before - len(df), rows_before, len(df))

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    df["Spam/Ham"] = df["Spam/Ham"].astype(str).str.strip().str.lower()

    rows_before_label_filter = len(df)
    df = df[df["Spam/Ham"].isin(["spam", "ham"])]
    if len(df) != rows_before_label_filter:
        logger.info("load_and_clean: dropped %d rows with invalid Spam/Ham label",
                    rows_before_label_filter - len(df))

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.reset_index(drop=True)

    if len(df) < 20:
        logger.error("load_and_clean: only %d usable rows after cleaning (need >= 20)", len(df))
        raise PipelineError(
            "Not enough usable rows after cleaning (need at least 20 rows with "
            "Spam/Ham equal to 'spam' or 'ham')."
        )
    if df["Spam/Ham"].nunique() < 2:
        logger.error("load_and_clean: only one class present after cleaning")
        raise PipelineError(
            "The file needs both spam and ham examples to train/evaluate a classifier."
        )

    logger.info("load_and_clean: cleaning complete, final shape=%s", df.shape)
    return df


def engineer_features(df):
    logger.info("ACTION engineer_features: computing text-derived numeric features for %d rows", len(df))
    df = df.copy()
    df["text"] = (df["Subject"].astype(str) + " " + df["Message"].astype(str)).str.strip()
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()
    df["avg_word_len"] = df["char_count"] / df["word_count"].replace(0, np.nan)
    df["capital_ratio"] = df["text"].apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1))
    df["digit_count"] = df["text"].apply(lambda t: sum(c.isdigit() for c in t))
    df["punct_count"] = df["text"].apply(lambda t: sum(1 for c in t if c in "!?.,;:$%"))
    df["exclaim_count"] = df["text"].apply(lambda t: t.count("!"))

    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            n_null = int(df[col].isnull().sum())
            df[col] = df[col].fillna(df[col].median())
            logger.info("engineer_features: filled %d nulls in %s with median", n_null, col)

    logger.info("engineer_features: done, columns added=%s", NUMERIC_COLS)
    return df

def _new_run_dir():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(RUN_DIR_ROOT, f"part2_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _fig_to_base64(fig, save_path=None):
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        logger.info("Saved chart to %s", save_path)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def load_cleaned(csv_path):
    logger.info("ACTION load_cleaned: reading %s", csv_path)
    if not os.path.exists(csv_path):
        raise PipelineError(f"cleaned_data.csv not found at {csv_path}. Run Part 1 first.")
    # keep_default_na=False: Part 1 deliberately stores "" (not NaN) for subject/body-empty
    # emails -- see Part 1 README's CSV serialization note. Re-reading with default NA
    # handling would reintroduce those as spurious nulls.
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
    required = {"word_count", "Spam/Ham", "Date", "digit_count", "avg_word_len",
                "capital_ratio", "punct_count", "exclaim_count"}
    missing = required - set(df.columns)
    if missing:
        raise PipelineError(f"cleaned_data.csv is missing required column(s): {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    logger.info("load_cleaned: shape=%s", df.shape)
    return df


def build_features_and_labels(df):
    """Builds X, y_reg, y_clf on a shared, leak-free feature matrix.

    Design notes (see README for full justification):
      - y_reg = word_count (continuous). y_clf = Spam/Ham binarized (spam=1/ham=0) --
        a natural existing binary column, used instead of median-binarizing y_reg.
      - char_count is EXCLUDED from X: it correlates with word_count at r=0.996
        (Part 1 finding) -- essentially the same measurement, so including it would be
        near-definitional leakage for the regression target.
      - Spam/Ham / Subject / Message / text / Date are excluded from X as raw/target
        columns; two features are instead *derived* from Date/digit_count specifically
        to have real categorical columns to encode:
          - weekday (nominal, no natural order)      -> one-hot, drop_first=True
          - digit_level (ordinal: Low<Medium<High)    -> label-encoded 0/1/2
    """
    logger.info("ACTION build_features_and_labels: engineering categorical features")
    df = df.copy()
    df["weekday"] = df["Date"].dt.day_name()
    df["digit_level"] = pd.cut(df["digit_count"], bins=[-1, 5, 20, np.inf],
                                labels=["Low", "Medium", "High"])

    y_reg = df["word_count"].astype(float)
    y_clf = (df["Spam/Ham"].astype(str).str.lower() == "spam").astype(int)

    # Ordinal label encoding -- order Low(0) < Medium(1) < High(2) reflects increasing
    # digit density, a real ordering (unlike e.g. city names), so integer encoding is valid.
    ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
    df["digit_level_enc"] = df["digit_level"].map(ordinal_map)

    # Nominal one-hot encoding -- weekday has no natural order, so label-encoding it would
    # falsely imply e.g. "Friday > Monday" to a linear/logistic model. drop_first=True avoids
    # the dummy-variable trap (perfect multicollinearity with the intercept).
    weekday_dummies = pd.get_dummies(df["weekday"], prefix="weekday", drop_first=True)

    X = pd.concat([df[BASE_NUMERIC_FEATURES], df[["digit_level_enc"]], weekday_dummies], axis=1)
    logger.info("build_features_and_labels: X shape=%s, columns=%s", X.shape, X.columns.tolist())
    return X, y_reg, y_clf


def split_and_scale(X, y_reg, y_clf, test_size=0.2, random_state=42):
    logger.info("ACTION split_and_scale: test_size=%.2f random_state=%d", test_size, random_state)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state
    )
    # Scaler fit on TRAIN ONLY. Fitting on the full dataset (train+test) would leak
    # test-set mean/variance into the training process -- the model would implicitly
    # "see" statistics about rows it's supposed to be evaluated on, inflating reported
    # performance relative to how the model would behave on truly unseen data.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    logger.info("split_and_scale: train=%d test=%d rows", len(X_train), len(X_test))
    return X_train_s, X_test_s, y_reg_train, y_reg_test, y_clf_train, y_clf_test, X.columns.tolist()


def run_regression(X_train_s, X_test_s, y_reg_train, y_reg_test, feature_names):
    logger.info("ACTION run_regression: fitting Linear + Ridge regression")
    lr = LinearRegression().fit(X_train_s, y_reg_train)
    pred_lr = lr.predict(X_test_s)
    mse_lr = mean_squared_error(y_reg_test, pred_lr)
    r2_lr = r2_score(y_reg_test, pred_lr)

    ridge = Ridge(alpha=1.0).fit(X_train_s, y_reg_train)
    pred_ridge = ridge.predict(X_test_s)
    mse_ridge = mean_squared_error(y_reg_test, pred_ridge)
    r2_ridge = r2_score(y_reg_test, pred_ridge)

    coef_table = (
        pd.DataFrame({"feature": feature_names, "lr_coef": lr.coef_, "ridge_coef": ridge.coef_})
        .assign(abs_lr_coef=lambda d: d["lr_coef"].abs())
        .sort_values("abs_lr_coef", ascending=False)
        .drop(columns="abs_lr_coef")
    )
    top3 = coef_table.head(3).to_dict(orient="records")

    logger.info("run_regression: OLS MSE=%.3f R2=%.4f | Ridge MSE=%.3f R2=%.4f",
                mse_lr, r2_lr, mse_ridge, r2_ridge)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(y_reg_test, pred_lr, alpha=0.15, s=10)
    lims = [0, np.quantile(y_reg_test, 0.99)]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual word_count"); ax.set_ylabel("Predicted word_count")
    ax.set_title(f"Linear Regression: Actual vs Predicted (R2={r2_lr:.3f})")

    return {
        "mse_lr": mse_lr, "r2_lr": r2_lr, "mse_ridge": mse_ridge, "r2_ridge": r2_ridge,
        "coef_table": coef_table, "top3": top3, "scatter_fig": fig,
    }


def check_class_balance(y_clf_train):
    logger.info("ACTION check_class_balance")
    vc = y_clf_train.value_counts()
    pct = (vc / len(y_clf_train) * 100).round(2)
    minority_pct = pct.min()
    needs_action = minority_pct < IMBALANCE_THRESHOLD_PCT
    logger.info("check_class_balance: counts=%s pct=%s minority=%.2f%% needs_action=%s",
                vc.to_dict(), pct.to_dict(), minority_pct, needs_action)
    return {"counts_before": vc.to_dict(), "pct_before": pct.to_dict(),
            "minority_pct": minority_pct, "needs_action": needs_action}


def run_classification(X_train_s, X_test_s, y_clf_train, y_clf_test, run_dir):
    logger.info("ACTION run_classification: fitting Logistic Regression (C=1.0)")
    balance_info = check_class_balance(y_clf_train)

    # class_weight='balanced' used regardless: costs nothing (a weighting scheme, not
    # resampling -- no synthetic rows, so "before"/"after" class counts are identical),
    # and provides margin if class balance drifts in future data. SMOTE was not applied
    # since the imbalance threshold (35%) wasn't crossed -- see README.
    logit = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    logit.fit(X_train_s, y_clf_train)
    pred = logit.predict(X_test_s)
    proba = logit.predict_proba(X_test_s)[:, 1]

    cm = confusion_matrix(y_clf_test, pred)
    report = classification_report(y_clf_test, pred, target_names=["ham", "spam"], output_dict=True)
    for cls in ["ham", "spam"]:
        report[cls]["support"] = int(report[cls]["support"])
    auc = roc_auc_score(y_clf_test, proba)
    fpr, tpr, _ = roc_curve(y_clf_test, proba)

    logger.info("run_classification: AUC=%.4f accuracy=%.4f", auc, report["accuracy"])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (C=1.0)")
    cm_chart = _fig_to_base64(fig, os.path.join(run_dir, "06_confusion_matrix_clf.png"))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Spam Classification (Part 2)")
    ax.annotate(f"AUC = {auc:.3f}", xy=(0.55, 0.15))
    ax.legend()
    roc_chart = _fig_to_base64(fig, os.path.join(run_dir, "07_roc_curve.png"))

    # Threshold sensitivity
    thresh_rows = []
    for t in THRESHOLDS:
        preds_t = (proba >= t).astype(int)
        thresh_rows.append({
            "threshold": t,
            "precision": precision_score(y_clf_test, preds_t, zero_division=0),
            "recall": recall_score(y_clf_test, preds_t, zero_division=0),
            "f1": f1_score(y_clf_test, preds_t, zero_division=0),
        })
    thresh_df = pd.DataFrame(thresh_rows)
    best_threshold = float(thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"])
    logger.info("run_classification: F1-maximizing threshold=%.2f", best_threshold)

    return {
        "balance_info": balance_info, "cm": cm, "report": report, "auc": auc,
        "cm_chart": cm_chart, "roc_chart": roc_chart, "thresh_df": thresh_df,
        "best_threshold": best_threshold, "proba": proba, "model": logit,
    }


def run_regularization_experiment(X_train_s, X_test_s, y_clf_train, y_clf_test, proba_c1):
    logger.info("ACTION run_regularization_experiment: fitting Logistic Regression (C=0.01)")
    logit_c01 = LogisticRegression(max_iter=1000, C=0.01, class_weight="balanced", random_state=42)
    logit_c01.fit(X_train_s, y_clf_train)
    pred_c01 = logit_c01.predict(X_test_s)
    proba_c01 = logit_c01.predict_proba(X_test_s)[:, 1]

    pred_c1 = (proba_c1 >= 0.5).astype(int)
    comparison = pd.DataFrame([
        {"model": "C=1.0", "precision": precision_score(y_clf_test, pred_c1),
         "recall": recall_score(y_clf_test, pred_c1), "auc": roc_auc_score(y_clf_test, proba_c1)},
        {"model": "C=0.01", "precision": precision_score(y_clf_test, pred_c01),
         "recall": recall_score(y_clf_test, pred_c01), "auc": roc_auc_score(y_clf_test, proba_c01)},
    ])
    logger.info("run_regularization_experiment: comparison=%s", comparison.to_dict(orient="records"))
    return {"comparison": comparison, "proba_c01": proba_c01}


def bootstrap_auc_diff(y_clf_test, proba_c1, proba_c01, n_boot=500, seed=42):
    logger.info("ACTION bootstrap_auc_diff: n_boot=%d", n_boot)
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_clf_test)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_arr), size=len(y_arr), replace=True)
        y_s = y_arr[idx]
        if len(np.unique(y_s)) < 2:
            continue
        auc1 = roc_auc_score(y_s, proba_c1[idx])
        auc01 = roc_auc_score(y_s, proba_c01[idx])
        diffs.append(auc1 - auc01)
    diffs = np.array(diffs)
    mean_diff = float(diffs.mean())
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    excludes_zero = bool(ci_low > 0 or ci_high < 0)
    logger.info("bootstrap_auc_diff: n_valid=%d mean=%.4f CI=[%.4f, %.4f] excludes_zero=%s",
                len(diffs), mean_diff, ci_low, ci_high, excludes_zero)
    return {"n_valid": len(diffs), "mean_diff": mean_diff, "ci_low": float(ci_low),
            "ci_high": float(ci_high), "excludes_zero": excludes_zero}

def run_pipeline(file_like):
    """End-to-end Part 2: cleaned_data.csv -> features/labels -> regression +
    classification models -> charts + tables, saved to a timestamped run dir."""

    logger.info("ACTION run_pipeline: starting full pipeline")
    run_dir = _new_run_dir()
    logger.info("run_pipeline: output directory for this run -> %s", run_dir)

    # Execute the complete Part 1 preprocessing pipeline:
    # - Clean the raw email dataset
    # - Remove duplicates and invalid records
    # - Handle missing values
    df = load_and_clean(file_like)
    
    # Generate engineered features required for ML models
    # (e.g., word_count, char_count, digit_count, punctuation count, etc.)
    df = engineer_features(df)

    # Create a timestamped output directory for the current execution
    run_dir = _new_run_dir()
    
    # Save the cleaned and feature-engineered dataset.
    # This file serves as the input dataset for Part 2 (Regression & Classification).
    cleaned_path = os.path.join(run_dir, "cleaned_data.csv")
    df.to_csv(cleaned_path, index=False)
    logger.info("run_pipeline: saved cleaned dataset -> %s (%d rows)", cleaned_path, len(df))

    # Load the cleaned dataset generated in Part 1.
    # Part 2 uses this processed dataset instead of the raw CSV
    # to ensure consistent preprocessing and feature engineering.
    df = load_cleaned(cleaned_path)
    
    X, y_reg, y_clf = build_features_and_labels(df)
    X_train_s, X_test_s, y_reg_train, y_reg_test, y_clf_train, y_clf_test, feat_names = \
        split_and_scale(X, y_reg, y_clf)

    reg_results = run_regression(X_train_s, X_test_s, y_reg_train, y_reg_test, feat_names)
    reg_scatter_b64 = _fig_to_base64(reg_results["scatter_fig"], os.path.join(run_dir, "05_regression_scatter.png"))

    clf_results = run_classification(X_train_s, X_test_s, y_clf_train, y_clf_test, run_dir)
    reg_experiment = run_regularization_experiment(
        X_train_s, X_test_s, y_clf_train, y_clf_test, clf_results["proba"])
    boot = bootstrap_auc_diff(y_clf_test, clf_results["proba"], reg_experiment["proba_c01"])

    logger.info("run_pipeline: complete, outputs saved to %s", run_dir)

    return {
        "cleaned_path": cleaned_path,
        "run_dir": run_dir,
        "feature_names": feat_names,
        "regression": {
            "mse_lr": reg_results["mse_lr"], "r2_lr": reg_results["r2_lr"],
            "mse_ridge": reg_results["mse_ridge"], "r2_ridge": reg_results["r2_ridge"],
            "coef_table": reg_results["coef_table"].to_dict(orient="records"),
            "top3": reg_results["top3"], "scatter_chart": reg_scatter_b64,
        },
        "classification": {
            "balance_info": clf_results["balance_info"],
            "cm": clf_results["cm"].tolist(),
            "report": clf_results["report"],
            "auc": clf_results["auc"],
            "cm_chart": clf_results["cm_chart"],
            "roc_chart": clf_results["roc_chart"],
            "thresh_table": clf_results["thresh_df"].to_dict(orient="records"),
            "best_threshold": clf_results["best_threshold"],
        },
        "regularization": {
            "comparison": reg_experiment["comparison"].to_dict(orient="records"),
        },
        "bootstrap": boot,
    }
