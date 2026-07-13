# Part 2 – Supervised Machine Learning

This module uses the cleaned dataset (`cleaned_data.csv`) generated in Part 1 to build and evaluate both Regression and Classification models.

---

## Dataset

- Source: `cleaned_data.csv`
- Records: 30,494
- Missing values handled in Part 1

---

## Target Variables

### Regression
Predict Word Count (`word_count`).

### Classification
Predict whether an email is:

- Spam (1)
- Ham (0)

---

## Features Used

Numeric Features

- avg_word_len
- capital_ratio
- digit_count
- punct_count
- exclaim_count

Encoded Features

- digit_level (Label Encoding)
- weekday (One-Hot Encoding)

Excluded Features

- Subject
- Message
- text
- Date
- Spam/Ham (target)
- char_count (highly correlated with word_count)

---

## Data Preparation

- Train/Test Split: 80% / 20%
- Random State: 42
- StandardScaler fitted only on training data to prevent data leakage.

---

# Regression Models

Models compared:

- Linear Regression
- Ridge Regression

### Performance

| Model | MSE | R² |
|------|------:|------:|
| Linear Regression | 102,513.36 | 0.8584 |
| Ridge Regression | 102,524.33 | 0.8584 |

### Important Features

1. punct_count
2. digit_count
3. avg_word_len

Both models performed almost identically because Ridge regularization had little impact on this dataset.

---

# Classification Model

Model Used

- Logistic Regression

### Class Distribution

| Class | Count |
|------|------:|
| Ham | 12,735 |
| Spam | 11,660 |

The dataset is reasonably balanced, so no SMOTE or oversampling was required.

---

## Classification Results

| Metric | Value |
|---------|-------|
| Accuracy | 70.5% |
| Precision | 70.2% |
| Recall | 66.8% |
| F1 Score | 68.5% |
| AUC | 0.7858 |

---

## Threshold Comparison

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|------|
| 0.30 | 0.601 | 0.910 | 0.724 |
| 0.40 | 0.651 | 0.790 | 0.714 |
| 0.50 | 0.702 | 0.668 | 0.685 |
| 0.60 | 0.751 | 0.515 | 0.611 |
| 0.70 | 0.869 | 0.370 | 0.519 |

Best threshold: 0.30

A lower threshold improves recall, making it more suitable for identifying spam emails in an archive.

---

## Regularization Comparison

| Model | Precision | Recall | AUC |
|------|-----------|--------|------|
| C = 1.0 | 0.702 | 0.668 | 0.7858 |
| C = 0.01 | 0.702 | 0.667 | 0.7837 |

Both models performed similarly, indicating minimal overfitting.

---

## Bootstrap Validation

500 bootstrap samples were used to compare AUC values.

- Mean Difference: 0.0021
- 95% Confidence Interval:
  - 0.0016 – 0.0026

Result: C = 1.0 performed slightly better, but the difference is practically insignificant.

---

## Generated Outputs

Each run saves:

```
part2_<timestamp>/

05_regression_scatter.png
06_confusion_matrix_clf.png
07_roc_curve.png
```

---

## Technologies Used

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Summary

- Clean dataset from Part 1
- Linear & Ridge Regression
- Logistic Regression
- Model evaluation
- Threshold tuning
- Regularization comparison
- Bootstrap validation
- Automatic report generation
