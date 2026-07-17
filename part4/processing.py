"""
part4_processing.py
Part 4 pipeline: LLM-powered feature, Track C -- Model Prediction Explanation.

Loads the tuned pipeline from Part 3 (best_model.pkl), runs .predict() /
.predict_proba() on hand-crafted feature-vector inputs, then asks an LLM to
produce a structured, schema-validated JSON explanation of each prediction.

TRACK CHOSEN: (C) Model Prediction Explanation Pipeline -- chosen because it's
the direct, natural extension of the model already built and serialized in
Part 3, rather than introducing a disconnected new dataset shape (Track A) or
re-scoring records against a rubric unrelated to the trained model (Track B).

HONESTY NOTE ON LLM CALLS: this module's LLM API domain is not reachable from
the sandbox this was developed in, and no API key is configured there, so the
actual call_llm() -> requests.post(...) path could not be exercised against a
real provider during development. Everything downstream of the API response
(JSON parsing, jsonschema validation, fallback handling, the PII guardrail,
and -- most importantly -- the actual model .predict()/.predict_proba() calls
against the real best_model.pkl) IS real and was tested end-to-end. When no
LLM_API_KEY is set, call_llm() returns a clearly-labeled MOCK response (see
_mock_llm_response()) so the full pipeline still runs top-to-bottom and every
other requirement (schema validation, guardrail, tables) can be demonstrated
honestly. Set LLM_API_KEY (and optionally LLM_API_URL / LLM_MODEL for a
non-OpenRouter-compatible provider) and re-run to get real model output.
"""
import io
import os
import re
import json
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import requests
import jsonschema

from logger_setup import get_logger

logger = get_logger()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "run"))
os.makedirs(RUN_DIR_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM API setup (RFC-agnostic: any provider accepting {model, messages} JSON
# over POST works, e.g. OpenRouter). Key is read from an environment
# variable -- never hardcoded.
# ---------------------------------------------------------------------------
LLM_API_URL = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
LLM_API_KEY_ENV = "LLM_API_KEY"

# ---------------------------------------------------------------------------
# Feature encoding -- MUST exactly mirror part2_processing.build_features_and_labels()
# so a hand-crafted record is encoded identically to how training data was.
# Column order captured directly from a real training run (see module test).
# ---------------------------------------------------------------------------
BASE_NUMERIC_FEATURES = ["avg_word_len", "capital_ratio", "digit_count", "punct_count", "exclaim_count"]
ORDINAL_MAP = {"Low": 0, "Medium": 1, "High": 2}
WEEKDAY_DUMMY_COLUMNS = [
    "weekday_Monday", "weekday_Saturday", "weekday_Sunday",
    "weekday_Thursday", "weekday_Tuesday", "weekday_Wednesday",
]  # "Friday" is the drop_first=True baseline -- all-zero row means Friday.
ALL_FEATURE_COLUMNS = BASE_NUMERIC_FEATURES + ["digit_level_enc"] + WEEKDAY_DUMMY_COLUMNS

# ---------------------------------------------------------------------------
# JSON schema for the LLM's explanation (Track C: >=5 required scalar fields)
# ---------------------------------------------------------------------------
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
}

FALLBACK_EXPLANATION = {k: None for k in EXPLANATION_SCHEMA["required"]}

SYSTEM_PROMPT = (
    "You are an assistant that explains a spam/ham email classifier's predictions "
    "to a non-technical support analyst. You will be given the model's input feature "
    "values, its predicted class, and its predicted probability for that class. "
    "Respond with ONLY a single valid JSON object (no markdown fences, no prose "
    "before or after) with exactly these fields: "
    '"prediction_label" (string: the predicted class), '
    '"confidence_level" (string: one of "low", "medium", "high", based on the '
    "predicted probability), "
    '"top_reason" (string: the single most likely feature-based reason for this '
    "prediction, referencing the actual feature values given), "
    '"second_reason" (string: a second contributing reason), '
    '"next_step" (string: a short recommended action for the analyst, e.g. '
    '"no action needed" or "flag for manual review"). '
    "Do not invent feature values that were not provided. Do not include any text "
    "outside the JSON object."
)

USER_PROMPT_TEMPLATE = (
    "Feature values:\n{feature_json}\n\n"
    "Model prediction: {predicted_label}\n"
    "Predicted probability for that class: {probability:.4f}\n\n"
    "Explain this prediction as instructed."
)


# ---------------------------------------------------------------------------
# PII guardrail (exact pattern specified by the assignment)
# ---------------------------------------------------------------------------
def has_pii(text):
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def _mock_llm_response(user_prompt, temperature):
    """Used only when LLM_API_KEY is not set, so the pipeline can still be
    demonstrated end-to-end. Clearly distinguishable from a real response --
    every field is prefixed "[MOCK]" and this function is never used when a
    real API key is configured."""
    logger.warning("call_llm: LLM_API_KEY not set -- returning a MOCK response, not a real LLM call")
    predicted = "spam" if "spam" in user_prompt.lower() else "ham"
    jitter = "" if temperature == 0 else " (mock jitter for temp=0.7)"
    return json.dumps({
        "prediction_label": f"[MOCK] {predicted}",
        "confidence_level": "medium",
        "top_reason": f"[MOCK] simulated reason, no live LLM_API_KEY configured{jitter}",
        "second_reason": "[MOCK] simulated secondary reason",
        "next_step": "[MOCK] set LLM_API_KEY and re-run for a real explanation",
    })


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512,
             api_url=None, api_key=None):
    """Reusable LLM call. Returns the assistant's raw text content, or None
    on a non-200 response. `api_url`/`api_key` can be passed explicitly (e.g.
    from a form in the web UI); if omitted, falls back to LLM_API_URL /
    LLM_API_KEY environment variables. Falls back to a labeled mock if no key
    is available from either source (see module docstring). The key is never
    logged or echoed back in any response/template."""
    api_url = api_url or LLM_API_URL
    api_key = api_key or os.environ.get(LLM_API_KEY_ENV)
    if not api_key:
        return _mock_llm_response(user_prompt, temperature)
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    logger.info("ACTION call_llm: url=%s model=%s temperature=%.2f", api_url, LLM_MODEL, temperature)
    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        logger.error("call_llm: non-200 response: %d %s", response.status_code, response.text[:300])
        print(f"LLM API error: status_code={response.status_code}")
        return None
    return response.json()["choices"][0]["message"]["content"]


def parse_and_validate(raw_response):
    """Strip/parse/validate an LLM response against EXPLANATION_SCHEMA.
    Returns (parsed_dict_or_fallback, status_str) where status_str is one of
    'pass', 'json_error', 'schema_error', 'no_response'."""
    if raw_response is None:
        return dict(FALLBACK_EXPLANATION), "no_response"

    text = raw_response.strip()
    # Some providers wrap JSON in markdown fences even when told not to;
    # strip those defensively without masking a genuine formatting failure.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("parse_and_validate: JSON decode error: %s", exc)
        print(f"JSON decode error: {exc}")
        return dict(FALLBACK_EXPLANATION), "json_error"

    try:
        jsonschema.validate(parsed, EXPLANATION_SCHEMA)
    except jsonschema.ValidationError as exc:
        logger.warning("parse_and_validate: schema validation error: %s", exc.message)
        print(f"Schema validation error: {exc.message}")
        return dict(FALLBACK_EXPLANATION), "schema_error"

    return parsed, "pass"


# ---------------------------------------------------------------------------
# Feature encoding + model loading
# ---------------------------------------------------------------------------
def encode_record(features):
    """Encode a hand-crafted feature dict into the exact column layout
    best_model.pkl was trained on. `features` must contain the 5 base
    numeric features plus 'digit_level' (Low/Medium/High) and 'weekday'
    (a day name; use 'Friday' for the drop_first baseline)."""
    row = {col: 0 for col in ALL_FEATURE_COLUMNS}
    for f in BASE_NUMERIC_FEATURES:
        row[f] = features[f]
    row["digit_level_enc"] = ORDINAL_MAP[features["digit_level"]]
    weekday_col = f"weekday_{features['weekday']}"
    if weekday_col in WEEKDAY_DUMMY_COLUMNS:
        row[weekday_col] = 1
    return pd.DataFrame([row], columns=ALL_FEATURE_COLUMNS)


def load_model(model_path):
    logger.info("ACTION load_model: %s", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run Part 3 first (or point this at an "
            "existing best_model.pkl)."
        )
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Hand-crafted demo inputs (3, as required)
# ---------------------------------------------------------------------------
SAMPLE_INPUTS = [
    {
        "label": "Digit-and-punctuation-heavy, Saturday",
        "features": {
            "avg_word_len": 4.1, "capital_ratio": 0.0, "digit_count": 42,
            "punct_count": 58, "exclaim_count": 3, "digit_level": "High", "weekday": "Saturday",
        },
    },
    {
        "label": "Low digit/punct density, midweek",
        "features": {
            "avg_word_len": 5.3, "capital_ratio": 0.0, "digit_count": 2,
            "punct_count": 9, "exclaim_count": 0, "digit_level": "Low", "weekday": "Tuesday",
        },
    },
    {
        "label": "Borderline / moderate signals",
        "features": {
            "avg_word_len": 4.7, "capital_ratio": 0.0, "digit_count": 12,
            "punct_count": 24, "exclaim_count": 1, "digit_level": "Medium", "weekday": "Friday",
        },
    },
]

# PII guardrail demo inputs (assignment requires >=2: one blocked, one clean)
PII_TEST_INPUTS = [
    ("Contact John at john.doe@example.com for details.", True),   # should be BLOCKED
    ("This message has moderate digit and punctuation counts.", False),  # should PROCEED
]


def confidence_from_proba(p):
    if p >= 0.85:
        return "high"
    if p >= 0.65:
        return "medium"
    return "low"


def run_pipeline(model_path, api_url=None, api_key=None):
    """api_url/api_key: optional, e.g. submitted via the web UI's form. Falls
    back to LLM_API_URL / LLM_API_KEY environment variables if not given, and
    to the labeled mock if neither source provides a key. The key is never
    included in the returned dict (so it can't end up rendered in a template)
    and never logged -- only whether a real key was used at all."""
    logger.info("ACTION run_pipeline: starting, model=%s, api_url=%s, using_form_key=%s", model_path, api_url or LLM_API_URL, bool(api_key))
    model = load_model(model_path)

    # --- call_llm smoke test (required demonstration) ---
    smoke_response = call_llm(
        "Reply with only the word: hello", "Reply with only the word: hello",
        temperature=0.0, max_tokens=10, api_url=api_url, api_key=api_key,
    )
    logger.info("run_pipeline: call_llm smoke test response=%r", smoke_response)

    # --- PII guardrail demonstration ---
    guardrail_results = []
    for text, expected_block in PII_TEST_INPUTS:
        blocked = has_pii(text)
        if blocked:
            print("Input blocked: PII detected.")
        guardrail_results.append({
            "input": text, "blocked": blocked, "expected_block": expected_block,
            "correct": blocked == expected_block,
        })
        logger.info("guardrail: input=%r blocked=%s expected=%s", text, blocked, expected_block)

    # --- Main pipeline: predict + explain for each of the 3 hand-crafted inputs ---
    rows = []
    temp_comparison_rows = []
    for sample in SAMPLE_INPUTS:
        feats = sample["features"]
        X = encode_record(feats)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        pred_idx = classes.index(pred)
        pred_proba = float(proba[pred_idx])
        predicted_label = "spam" if pred == 1 else "ham"

        feature_json = json.dumps(feats)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            feature_json=feature_json, predicted_label=predicted_label, probability=pred_proba
        )

        # PII guardrail applied before every LLM call, per spec.
        if has_pii(user_prompt):
            print("Input blocked: PII detected.")
            explanation, status = dict(FALLBACK_EXPLANATION), "blocked"
            raw_response = None
        else:
            raw_response = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=300,
                                     api_url=api_url, api_key=api_key)
            explanation, status = parse_and_validate(raw_response)

        print(f"--- {sample['label']} ---")
        print("Input features:", feats)
        print("Predicted class:", predicted_label, "probability:", pred_proba)
        print("Raw LLM response:", raw_response)
        print("Validation status:", status)

        rows.append({
            "label": sample["label"], "features": feats, "predicted_label": predicted_label,
            "probability": pred_proba, "raw_response": raw_response,
            "explanation": explanation, "status": status,
        })

        # --- Temperature A/B comparison (temp=0 vs temp=0.7) for this input ---
        resp_t0 = raw_response  # already have temp=0 result from above
        raw_t07 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7, max_tokens=300,
                            api_url=api_url, api_key=api_key)
        temp_comparison_rows.append({
            "label": sample["label"], "output_t0": resp_t0, "output_t07": raw_t07,
        })

    logger.info("run_pipeline: complete")
    return {
        "smoke_test_response": smoke_response,
        "guardrail_results": guardrail_results,
        "rows": rows,
        "temp_comparison_rows": temp_comparison_rows,
        "using_mock": not bool(api_key or os.environ.get(LLM_API_KEY_ENV)),
        "api_url_used": api_url or LLM_API_URL,
    }
