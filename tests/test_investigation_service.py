"""
Tests for src/services/investigation_service.py

These tests cover the service layer independently of the ML models.
All tests that do not need real model artifacts use a patched
default_classifier so they are fast, deterministic, and CI-safe.

Test marks:
  unit        — no model artifacts needed (mocked classifier)
  integration — requires models/ artifacts on disk
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.investigation_service import analyze_review, get_model_status

# Real dataclasses used to construct valid predict() return values
from src.model.classifier import PredictionResult, SinglePrediction


# ---------------------------------------------------------------------------
# Helpers — build a minimal mock GhostSignalClassifier result
# ---------------------------------------------------------------------------

def _make_single_prediction(
    label: str,
    proba: dict[str, float],
    caveat: str = "Test caveat.",
) -> SinglePrediction:
    """Return a real SinglePrediction dataclass so dataclasses.asdict() works."""
    return SinglePrediction(
        label=label,
        probabilities=proba,
        is_uncertain=max(proba.values()) < 0.60,
        model_available=True,
        caveat=caveat,
        error=None,
    )


def _make_prediction_result(
    sent_label: str = "Positive",
    sent_proba: dict[str, float] | None = None,
    auth_label: str = "HONEST",
    auth_proba: dict[str, float] | None = None,
    text_preview: str = "mock text preview",
) -> PredictionResult:
    """Return a real PredictionResult dataclass so dataclasses.asdict() works."""
    if sent_proba is None:
        sent_proba = {"Negative": 0.05, "Positive": 0.95}
    if auth_proba is None:
        auth_proba = {"FAKE": 0.20, "HONEST": 0.80}
    return PredictionResult(
        text_preview=text_preview,
        sentiment=_make_single_prediction(sent_label, sent_proba, "Sentiment caveat."),
        authenticity=_make_single_prediction(auth_label, auth_proba, "Authenticity caveat."),
    )


# ---------------------------------------------------------------------------
# UNIT TESTS — mocked classifier, no disk I/O
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_empty_string_returns_ok_false(mock_clf):
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("")
    assert result["ok"] is False
    assert result["error_message"] is not None
    assert len(result["error_message"]) > 0


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_whitespace_only_returns_ok_false(mock_clf):
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("   \n\t  ")
    assert result["ok"] is False


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_empty_has_all_required_keys(mock_clf):
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("")
    for key in ("ok", "error_message", "text_preview", "models_ready",
                "sentiment", "authenticity", "is_short_text"):
        assert key in result, f"Missing key: {key!r}"


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_empty_is_short_text_is_false(mock_clf):
    """Empty input returns is_short_text=False (no meaningful word count)."""
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("")
    assert result["is_short_text"] is False


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_returns_all_required_keys(mock_clf):
    mock_clf.predict.return_value = _make_prediction_result()
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("This product is great!")
    for key in ("ok", "error_message", "text_preview", "models_ready",
                "sentiment", "authenticity", "is_short_text"):
        assert key in result, f"Missing key: {key!r}"


# ---------------------------------------------------------------------------
# is_short_text behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_is_short_text_false_for_five_words(mock_clf):
    """Exactly 5 words: NOT short (threshold is strictly < 5)."""
    mock_clf.predict.return_value = _make_prediction_result()
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("one two three four five")
    assert result["is_short_text"] is False


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_is_short_text_false_for_long_review(mock_clf):
    mock_clf.predict.return_value = _make_prediction_result()
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    text = "This product is absolutely fantastic. I would highly recommend it to everyone."
    result = analyze_review(text)
    assert result["is_short_text"] is False


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_is_short_text_true_for_one_word(mock_clf):
    mock_clf.predict.return_value = _make_prediction_result()
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("Good.")
    assert result["is_short_text"] is True


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_is_short_text_true_for_four_words(mock_clf):
    mock_clf.predict.return_value = _make_prediction_result()
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("one two three four")
    assert result["is_short_text"] is True


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_is_short_text_does_not_alter_predictions(mock_clf):
    """is_short_text must not change label, probabilities, or is_uncertain."""
    mock_clf.predict.return_value = _make_prediction_result(
        sent_label="Positive",
        sent_proba={"Negative": 0.0001, "Positive": 0.9999},
    )
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}

    result = analyze_review("Good.")
    assert result["is_short_text"] is True
    # Predictions must pass through unchanged
    assert result["sentiment"]["label"] == "Positive"
    assert result["sentiment"]["probabilities"]["Positive"] == pytest.approx(0.9999)


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_short_input_still_returns_ok_true_and_predictions(mock_clf):
    """
    A valid input with fewer than 5 words must still produce ok=True and
    both sentiment and authenticity predictions.  is_short_text=True is a
    display hint only — it must never suppress the prediction result.
    """
    mock_clf.predict.return_value = _make_prediction_result(
        sent_label="Positive",
        sent_proba={"Negative": 0.0001, "Positive": 0.9999},
        auth_label="HONEST",
        auth_proba={"FAKE": 0.10, "HONEST": 0.90},
    )
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}

    result = analyze_review("Good.")  # 1 word — below threshold

    assert result["ok"] is True
    assert result["is_short_text"] is True
    # Both prediction dicts must be present and non-None
    assert result["sentiment"] is not None
    assert result["authenticity"] is not None
    # Labels must be the values from the mock, not suppressed
    assert result["sentiment"]["label"] == "Positive"
    assert result["authenticity"]["label"] == "HONEST"
    # Probabilities must be intact
    assert result["sentiment"]["probabilities"]["Positive"] == pytest.approx(0.9999)
    assert result["authenticity"]["probabilities"]["HONEST"] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# ok=True path — prediction passthrough
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_ok_true_for_valid_text(mock_clf):
    mock_clf.predict.return_value = _make_prediction_result()
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    result = analyze_review("This is a great product, really happy with it.")
    assert result["ok"] is True
    assert result["error_message"] is None


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_preserves_sentiment_fields(mock_clf):
    mock_clf.predict.return_value = _make_prediction_result(
        sent_label="Negative",
        sent_proba={"Negative": 0.92, "Positive": 0.08},
    )
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}

    result = analyze_review("Terrible product, completely broken.")
    assert result["ok"] is True
    s = result["sentiment"]
    assert s["label"] == "Negative"
    assert s["probabilities"]["Negative"] == pytest.approx(0.92)
    assert s["is_uncertain"] is False
    assert s["model_available"] is True


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_preserves_authenticity_fields(mock_clf):
    mock_clf.predict.return_value = _make_prediction_result(
        auth_label="FAKE",
        auth_proba={"FAKE": 0.85, "HONEST": 0.15},
    )
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}

    result = analyze_review("Buy this now! Best product ever!")
    a = result["authenticity"]
    assert a["label"] == "FAKE"
    assert a["probabilities"]["FAKE"] == pytest.approx(0.85)


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_models_ready_included(mock_clf):
    mock_clf.predict.return_value = _make_prediction_result()
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": False}
    result = analyze_review("Some review text here.")
    assert "models_ready" in result
    assert isinstance(result["models_ready"], dict)


# ---------------------------------------------------------------------------
# Error-path: classification raises
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_handles_predict_exception(mock_clf):
    mock_clf.predict.side_effect = RuntimeError("model broken")
    mock_clf.is_ready.return_value = {"sentiment": False, "authenticity": False}

    result = analyze_review("Some valid text here.")
    assert result["ok"] is False
    assert result["error_message"] is not None
    # Must not expose internal exception detail
    assert "RuntimeError" not in result["error_message"]
    assert "model broken" not in result["error_message"]
    # is_short_text must still be present
    assert "is_short_text" in result


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_error_path_has_all_keys(mock_clf):
    mock_clf.predict.side_effect = ValueError("boom")
    mock_clf.is_ready.return_value = {"sentiment": False, "authenticity": False}
    result = analyze_review("Test review text.")
    for key in ("ok", "error_message", "text_preview", "models_ready",
                "sentiment", "authenticity", "is_short_text"):
        assert key in result, f"Missing key on error path: {key!r}"


@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_analyze_review_ok_true_when_is_ready_raises(mock_clf):
    """
    predict() succeeds but is_ready() raises an exception.

    analyze_review() must still return a valid dict with ok=True and the
    prediction data intact.  The models_ready fallback must be
    {"sentiment": False, "authenticity": False} — provided by
    _safe_model_status() — rather than propagating the exception.
    """
    mock_clf.predict.return_value = _make_prediction_result()
    mock_clf.is_ready.side_effect = RuntimeError("status check failed")

    result = analyze_review("This product is really great value for money.")

    assert isinstance(result, dict), "Result must be a dict"
    assert result["ok"] is True, "ok must be True when predict() succeeds"
    assert result["error_message"] is None
    # Prediction data must be present
    assert result["sentiment"] is not None
    assert result["authenticity"] is not None
    # _safe_model_status() must catch the is_ready() exception and return
    # the documented fallback rather than letting the exception propagate.
    assert result["models_ready"] == {"sentiment": False, "authenticity": False}, (
        f"Expected safe fallback for models_ready, got {result['models_ready']!r}"
    )


# ---------------------------------------------------------------------------
# get_model_status
# ---------------------------------------------------------------------------

@pytest.mark.unit
@patch("src.services.investigation_service.default_classifier")
def test_get_model_status_returns_dict_with_expected_keys(mock_clf):
    mock_clf.is_ready.return_value = {"sentiment": True, "authenticity": True}
    status = get_model_status()
    assert "sentiment" in status
    assert "authenticity" in status


# ---------------------------------------------------------------------------
# INTEGRATION TESTS — require saved model artifacts
# ---------------------------------------------------------------------------

from src.model.classifier import SENTIMENT_PATH, AUTHENTICITY_PATH
_ARTIFACTS_AVAILABLE = SENTIMENT_PATH.exists() and AUTHENTICITY_PATH.exists()
_SKIP = pytest.mark.skipif(
    not _ARTIFACTS_AVAILABLE,
    reason="Model artifacts not found — run python -m src.model.train first.",
)


@_SKIP
@pytest.mark.integration
def test_integration_long_review_is_not_short_text():
    result = analyze_review(
        "This product exceeded all my expectations. "
        "The build quality is excellent and it arrived quickly."
    )
    assert result["ok"] is True
    assert result["is_short_text"] is False


@_SKIP
@pytest.mark.integration
def test_integration_single_word_is_short_text():
    result = analyze_review("Good.")
    assert result["ok"] is True
    assert result["is_short_text"] is True
    # Prediction must still be present and valid
    assert result["sentiment"]["label"] in ("Positive", "Negative", "uncertain")


@_SKIP
@pytest.mark.integration
def test_integration_four_words_is_short_text():
    result = analyze_review("Love it very much")
    assert result["is_short_text"] is True


@_SKIP
@pytest.mark.integration
def test_integration_five_words_is_not_short_text():
    result = analyze_review("I really love this product")
    assert result["is_short_text"] is False


@_SKIP
@pytest.mark.integration
def test_integration_short_text_does_not_suppress_prediction():
    """is_short_text=True must not change the label or probabilities."""
    result = analyze_review("Good.")
    assert result["ok"] is True
    s = result["sentiment"]
    assert s["label"] in ("Positive", "Negative", "uncertain", "unavailable")
    assert isinstance(s["probabilities"], dict)
    if s["probabilities"]:
        total = sum(s["probabilities"].values())
        assert abs(total - 1.0) < 0.01


@_SKIP
@pytest.mark.integration
def test_integration_caveat_contains_binary_note():
    """Sentiment caveat must mention binary classification limitation."""
    result = analyze_review("Average product, nothing special.")
    caveat = result["sentiment"]["caveat"]
    assert "Neutral" in caveat or "two classes" in caveat or "binary" in caveat.lower()


@_SKIP
@pytest.mark.integration
def test_integration_auth_caveat_mentions_promotional():
    """Authenticity caveat must mention promotional language limitation."""
    result = analyze_review("Great product, would recommend!")
    caveat = result["authenticity"]["caveat"]
    assert "promotional" in caveat.lower() or "marketing" in caveat.lower()
