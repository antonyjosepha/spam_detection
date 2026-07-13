# Spam Detection Web Application

A web-based spam detection application built with **CherryPy**, **Genshi**, and **jQuery**. Upload an email dataset in CSV format to clean the data, generate insights, train a spam classifier, and view the results.

---

## Features

- Upload email CSV files
- Automatic data cleaning
- Feature engineering
- Spam classification using **TF-IDF + Naive Bayes**
- Interactive dashboard
- Charts and statistics
- Prediction results
- Automatic logging
- Saves processed data and charts

---

## Installation

```bash
pip install -r requirements.txt
python3.8 server.py
```

Open your browser and visit:

```
http://localhost:8080
```

---

## Expected CSV Format

The uploaded CSV should contain:

| Column | Description |
|---------|-------------|
| Subject | Email subject |
| Message | Email body |
| Spam/Ham | `spam` or `ham` |
| Date | Optional |

---

## Processing Pipeline

The application performs the following steps:

1. Clean the dataset
   - Remove duplicate emails
   - Fill missing Subject and Message
   - Keep only Spam/Ham records

2. Generate Features
   - Word Count
   - Character Count
   - Average Word Length
   - Capital Letter Ratio
   - Digit Count
   - Punctuation Count
   - Exclamation Count

3. Train Model
   - TF-IDF Vectorizer
   - Multinomial Naive Bayes
   - 75% Training / 25% Testing

4. Display Results
   - Model Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix
   - Statistics
   - Charts
   - Predictions

---

## Output Files

For every upload, the application creates a timestamped folder under:

```
../run/

```

Example:

```
../run/part1_20260713_182839_140523

├── cleaned_data.csv
├── 01_bar_chart.png
├── 02_histogram.png
├── 03_scatter_plot.png
├── 04_correlation_heatmap.png
└── 05_confusion_matrix.png
```

---

## Project Structure

```
server.py              Web server
processing.py          Data processing and ML pipeline
logger_setup.py        Logging
templates/             HTML templates
static/                CSS and JavaScript
logs/                  Daily log files
run/                   Generated outputs
```

---

## Logging

Logs are stored in:

```
logs/spam_detection_part1_YYYYMMDD.log
```

The application logs:

- User requests
- File uploads
- Data cleaning
- Feature generation
- Model training
- Performance metrics
- Errors and warnings

---

## Technologies Used

- Python
- CherryPy
- Genshi
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- jQuery

---

## Model

- TF-IDF Vectorizer
- Multinomial Naive Bayes Classifier

---

## Notes

- Maximum upload size: **300 MB**
- Charts are generated automatically and displayed in the browser.
- Processed files and visualizations are saved for future reference.
