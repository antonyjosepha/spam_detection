# Part 3 – Advanced Spam Detection Model

## Overview

In this part, multiple machine learning models were trained and compared to improve spam detection accuracy.

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Hyperparameter tuning was performed using **GridSearchCV** to find the best Random Forest model.

---

## Features

- Decision Tree with overfitting analysis
- Gini vs Entropy comparison
- Random Forest feature importance
- Gradient Boosting model
- Feature selection
- 5-Fold Cross Validation
- Hyperparameter tuning using GridSearchCV
- Learning Curve analysis
- Model serialization using Joblib

---

## Best Model

The best performing model is the **Tuned Random Forest Pipeline**.

**Parameters**

- n_estimators = 200
- max_depth = None
- min_samples_leaf = 5

**Performance**

| Metric | Score |
|---------|------:|
| Test AUC | **0.8855** |
| CV AUC | **0.8833** |

The model is saved as:

```
best_model.pkl
```

---

## Run

```bash
python server.py
```

Open:

```
http://localhost:8080/part3
```

---

## Project Files

```
processing.py
server.py
templates/
README.md
run/best_model.pkl
```

---

## Notes

- The tuned Random Forest model provides the best performance.
- The trained model is saved using Joblib.
- The saved model can be loaded directly without retraining.

Example:

```python
import joblib

model = joblib.load("best_model.pkl")
prediction = model.predict(X_test)
```
