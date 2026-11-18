"""
GhostSignal — Investigation Service
=====================================
Provides the application-level bridge between the Streamlit frontend
and the GhostSignalClassifier ML backend.

Public surface
--------------
  analyze_review(text: str) -> dict   — classify a single review text
  get_model_status() -> dict          — check whether both pipelines are loaded

Design notes
------------
- Uses the module-level ``default_classifier`` instance so the .joblib
  pipelines are loaded once per process and reused across Streamlit reruns.
- Input validation is performed here (not in the classifier) so the
  frontend receives a consistent error dict rather than an exception.
- All return values are plain Python dicts (JSON-serialisable) so the
  frontend never needs to import ML types directly.
- The classifier's existing predictions, probabilities, uncertainty flags
  and caveats are preserved verbatim — this layer does not alter them.
"""

import dataclasses
import logging
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared classifier instance
# ---------------------------------------------------------------------------
# Imported here so the .joblib pipelines are loaded the first time
# analyze_review() is called and then cached for the process lifetime.

from src.model.classifier import default_classifier  # noqa: E402


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_status() -> dict[str, bool]:
    """
    Return the load status of both ML pipelines.

    Returns
    -------
    dict with keys:
        "sentiment"     : True if the sentiment pipeline loaded successfully
        "authenticity"  : True if the authenticity pipeline loaded successfully
    """
    return default_classifier.is_ready()


def _safe_model_status() -> dict[str, bool]:
    """
    Wrapper around get_model_status() that never raises.

    If get_model_status() itself throws (e.g. a joblib deserialization error
    on first access), this returns a safe fallback dict so that
    analyze_review() can still return a structured error dict instead of
    propagating an unhandled exception.
    """
    try:
        return get_model_status()
    except Exception as exc:
        log.error("get_model_status() raised unexpectedly: %s", exc, exc_info=True)
        return {"sentiment": False, "authenticity": False}


def analyze_review(text: str) -> dict[str, Any]:
    """
    Classify a single review text with both the sentiment and authenticity
    classifiers and return a plain dict suitable for direct rendering in the
    Streamlit frontend.

    Parameters
    ----------
    text : str
        Raw review text entered by the user.  May be empty.

    Returns
    -------
    dict with the following guaranteed keys:

        "ok"            : bool  — False on validation failure or unexpected error
        "error_message" : str | None — human-readable error when ok=False
        "text_preview"  : str  — first 120 chars of the input (for display)
        "models_ready"  : dict — {"sentiment": bool, "authenticity": bool}
        "sentiment"     : dict — SinglePrediction fields (see below), or None
        "authenticity"  : dict — SinglePrediction fields (see below), or None
        "is_short_text" : bool — True when the input contains fewer than 5 words.
                                 When True the UI should warn that the model has
                                 limited text to work with and that the probability
                                 scores may appear extreme due to very few active
                                 features.  Predictions and probabilities are
                                 unchanged regardless of this flag.

    Each classifier sub-dict contains:
        "label"           : str   — predicted class, "uncertain", or "unavailable"
        "probabilities"   : dict  — {class_name: float probability score}
        "is_uncertain"    : bool  — True when max probability < 0.60
        "model_available" : bool  — False when the .joblib file was not found
        "caveat"          : str   — fixed disclaimer about label meaning
        "error"           : str | None — set when input is invalid or a
                                         runtime error occurred

    Notes
    -----
    - Probabilities are model scores, not calibrated confidence values.
    - An "uncertain" label means the model's winning class probability is
      below 0.60; it does not mean the review is safe, genuine, or fake.
    - A "FAKE" authenticity label means the text resembles writing patterns
      of reviewers labeled fake in the training dataset.  It does NOT prove
      the submitted review is fraudulent.
    - is_short_text is a display hint only.  It never alters predictions,
      probabilities, labels, or any classifier output.
    - This function never raises; all failures are captured into the dict.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(text, str) or not text.strip():
        return {
            "ok": False,
            "error_message": "Please enter some review text before analyzing.",
            "text_preview": "",
            "models_ready": _safe_model_status(),
            "sentiment": None,
            "authenticity": None,
            "is_short_text": False,
        }

    # ------------------------------------------------------------------
    # Short-text detection (display hint only — does not affect predictions)
    # ------------------------------------------------------------------
    # Fewer than 5 words means very few TF-IDF features fire, which can
    # produce extreme-looking probability scores from just one or two
    # high-magnitude vocabulary coefficients.  The prediction is not
    # suppressed or adjusted; the flag is exposed so the UI can warn the user.
    _SHORT_TEXT_WORD_THRESHOLD = 5
    is_short_text: bool = len(text.split()) < _SHORT_TEXT_WORD_THRESHOLD

    # ------------------------------------------------------------------
    # Classification and dataclass conversion
    # ------------------------------------------------------------------
    # Both steps are inside one try/except so that an unexpected failure
    # in either predict() or dataclasses.asdict() is caught and returned
    # as a structured error dict rather than propagating to the caller.
    try:
        result = default_classifier.predict(text)
        # dataclasses.asdict() performs a deep copy, converting all nested
        # dataclasses to dicts.  PredictionResult contains two SinglePrediction
        # instances; the result is fully JSON-serialisable.
        result_dict = dataclasses.asdict(result)
    except Exception as exc:
        log.error("Unexpected error during review analysis: %s", exc, exc_info=True)
        return {
            "ok": False,
            "error_message": (
                "Analysis could not be completed due to an internal error. "
                "Please try again later."
            ),
            "text_preview": text[:120],
            "models_ready": _safe_model_status(),
            "sentiment": None,
            "authenticity": None,
            "is_short_text": is_short_text,
        }

    return {
        "ok": True,
        "error_message": None,
        "text_preview": result_dict["text_preview"],
        "models_ready": _safe_model_status(),
        "sentiment": result_dict["sentiment"],
        "authenticity": result_dict["authenticity"],
        "is_short_text": is_short_text,
    }
