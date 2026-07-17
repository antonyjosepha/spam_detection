# Spam Detection - Part 4 (LLM Prediction Explanation)

## Overview

Part 4 extends the Spam Detection project by generating an AI explanation for each model prediction using an LLM.

The application:
- Loads the trained spam detection model (`spam_model.joblib`)
- Predicts whether an email is Spam or Ham
- Uses an LLM to explain the prediction
- Validates the LLM response
- Blocks requests containing PII (email addresses or phone numbers)

---

## Prerequisites

- Python 3.8+
- Install the required packages

```bash
pip install -r requirements.txt
```

---

## Before Running Part 4

Update the **API Key** in the `meta.ini` file.

Example:

```ini
[llm]
api_url=https://openrouter.ai/api/v1/chat/completions
api_key=
model_path=best_model.pkl
```

Without a valid API key, the application will return mock responses.

---

## Run the Application

```bash
python server.py
```

Open your browser:

```
http://localhost:8080/part4
```

---

## Files

- `server.py` - Cherrypy web application
- `processing.py` - Prediction and LLM pipeline
- `spam_model.joblib` - Trained spam detection model
- `meta.ini` - LLM configuration
- `templates/` - HTML templates

---

## Features

- Spam/Ham prediction
- AI-generated explanation
- JSON schema validation
- PII detection
- Confidence score
- Fallback response if LLM is unavailable

---

## Notes

- Ensure `spam_model.joblib` is available before starting the application.
- Update the API key in `meta.ini` before running Part 4.
- Internet access is required for real LLM responses.
