"""
processing.py
Core pipeline for the Spam Detection web app.

Reuses the cleaning / feature-engineering logic from the offline EDA scripts
(01_load_clean_dtypes.py) and adds a TF-IDF + Multinomial Naive Bayes
classifier so an uploaded CSV can be evaluated end-to-end in one request.
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
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from logger_setup import get_logger

logger = get_logger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Charts and cleaned_data.csv are written here per upload, as requested:
# a sibling "tmp" directory one level up from the app folder.
TMP_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "tmp"))
os.makedirs(TMP_DIR, exist_ok=True)

RUN_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "run"))
os.makedirs(RUN_DIR, exist_ok=True)

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
    """Create a fresh timestamped subfolder under ../tmp for this upload's outputs."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(RUN_DIR, f"part1_{run_id}")
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


def make_charts(df, run_dir):
    logger.info("ACTION make_charts: generating bar/hist/scatter/heatmap charts")
    charts = {}
    palette = {"spam": "#C0504D", "ham": "#1F4E79"}

    fig, ax = plt.subplots(figsize=(6, 4.3))
    df.groupby("Spam/Ham")["word_count"].mean().reindex(["ham", "spam"]).plot.bar(
        ax=ax, color=[palette["ham"], palette["spam"]]
    )
    ax.set_title("Mean Word Count by Class")
    ax.set_ylabel("Mean word count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    charts["bar"] = _fig_to_base64(fig, os.path.join(run_dir, "01_bar_chart.png"))

    fig, ax = plt.subplots(figsize=(6, 4.3))
    sns.histplot(df["char_count"].clip(upper=df["char_count"].quantile(0.99)),
                 bins=25, ax=ax, color="#1F4E79")
    ax.set_title("Character Count Distribution (99th pct clipped)")
    ax.set_xlabel("char_count")
    charts["hist"] = _fig_to_base64(fig, os.path.join(run_dir, "02_histogram.png"))

    fig, ax = plt.subplots(figsize=(6, 4.3))
    sample = df.sample(min(2000, len(df)), random_state=1)
    sns.scatterplot(data=sample, x="word_count", y="char_count", hue="Spam/Ham",
                     alpha=0.4, s=15, ax=ax, palette=palette)
    ax.set_title("Word Count vs Character Count")
    charts["scatter"] = _fig_to_base64(fig, os.path.join(run_dir, "03_scatter_plot.png"))

    fig, ax = plt.subplots(figsize=(6.3, 5))
    corr = df[NUMERIC_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heat Map")
    charts["heatmap"] = _fig_to_base64(fig, os.path.join(run_dir, "04_correlation_heatmap.png"))

    logger.info("make_charts: generated %d charts, saved to %s", len(charts), run_dir)
    return charts


def confusion_matrix_chart(cm, run_dir):
    logger.info("ACTION confusion_matrix_chart: rendering confusion matrix image")
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test split)")
    return _fig_to_base64(fig, os.path.join(run_dir, "05_confusion_matrix.png"))


def train_and_evaluate(df, test_size=0.25, random_state=42):
    logger.info("ACTION train_and_evaluate: training TF-IDF + MultinomialNB on %d rows (test_size=%.2f)",
                len(df), test_size)
    X = df["text"]
    y = df["Spam/Ham"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("train_and_evaluate: split into train=%d test=%d rows", len(X_train), len(X_test))

    vectorizer = TfidfVectorizer(max_features=8000, stop_words="english", min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    logger.info("train_and_evaluate: TF-IDF vocabulary size=%d", len(vectorizer.vocabulary_))

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)

    classes = list(model.classes_)
    spam_idx = classes.index("spam") if "spam" in classes else 0

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, preds, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, preds, pos_label="spam", zero_division=0),
    }
    cm = confusion_matrix(y_test, preds, labels=["ham", "spam"])

    results_df = df.loc[idx_test, ["Subject"]].copy()
    results_df["actual"] = y_test.values
    results_df["predicted"] = preds
    results_df["spam_probability"] = probs[:, spam_idx]
    results_df["correct"] = results_df["actual"] == results_df["predicted"]
    results_df = results_df.sort_values("spam_probability", ascending=False).reset_index(drop=True)

    logger.info(
        "train_and_evaluate: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
    )
    return metrics, cm, results_df


def run_pipeline(file_like):
    """End-to-end: raw CSV -> cleaned df -> features -> charts -> model results.
    Saves charts (PNG) and cleaned_data.csv to a timestamped folder under
    ../tmp, and returns a dict ready to hand to the results template.
    """
    logger.info("ACTION run_pipeline: starting full pipeline")
    run_dir = _new_run_dir()
    logger.info("run_pipeline: output directory for this run -> %s", run_dir)

    df = load_and_clean(file_like)
    df = engineer_features(df)

    cleaned_path = os.path.join(run_dir, "cleaned_data.csv")
    df.to_csv(cleaned_path, index=False)
    logger.info("run_pipeline: saved cleaned dataset -> %s (%d rows)", cleaned_path, len(df))

    charts = make_charts(df, run_dir)
    metrics, cm, results_df = train_and_evaluate(df)
    cm_chart = confusion_matrix_chart(cm, run_dir)

    metrics_display = {
        "accuracy": f"{metrics['accuracy'] * 100:.1f}%",
        "precision": f"{metrics['precision'] * 100:.1f}%",
        "recall": f"{metrics['recall'] * 100:.1f}%",
        "f1": f"{metrics['f1'] * 100:.1f}%",
    }

    desc = df[NUMERIC_COLS].describe().round(2)
    stats_rows = []
    for stat_name, row in desc.iterrows():
        r = {"stat": stat_name}
        r.update({k: f"{v:.2f}" for k, v in row.to_dict().items()})
        stats_rows.append(r)

    results = []
    for r in results_df.head(300).to_dict(orient="records"):
        subj = str(r["Subject"])
        results.append({
            "subject_short": (subj[:90] + "…") if len(subj) > 90 else (subj or "(no subject)"),
            "actual": r["actual"],
            "predicted": r["predicted"],
            "spam_prob_display": f"{r['spam_probability']:.3f}",
            "correct": bool(r["correct"]),
            "row_class": "row-correct" if r["correct"] else "row-wrong",
        })

    summary = {
        "n_rows": len(df),
        "n_spam": int((df["Spam/Ham"] == "spam").sum()),
        "n_ham": int((df["Spam/Ham"] == "ham").sum()),
        "output_dir": run_dir,
        "cleaned_data_path": cleaned_path,
    }

    logger.info("run_pipeline: complete, rows=%d spam=%d ham=%d, outputs saved to %s",
                summary["n_rows"], summary["n_spam"], summary["n_ham"], run_dir)

    return {
        "summary": summary,
        "metrics": metrics_display,
        "charts": charts,
        "cm_chart": cm_chart,
        "numeric_cols": NUMERIC_COLS,
        "stats_rows": stats_rows,
        "results": results,
    }
