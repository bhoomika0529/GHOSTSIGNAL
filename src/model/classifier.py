"""
GhostSignal — Prediction Interface
=====================================
Loads the trained TF-IDF + Logistic Regression pipelines and exposes a
clean prediction API consumed by the backend / investigation service.

Artifact paths
--------------
  models/sentiment_pipeline.joblib     — sentiment classifier
  models/authenticity_pipeline.joblib  — authenticity classifier

Usage
-----
    from src.model.classifier import GhostSignalClassifier

    clf = GhostSignalClassifier()
    result = clf.predict("This product is absolutely terrible, broke in a week.")
    # result is a dict — see predict() docstring for schema

Important caveats (enforced in docstrings and output)
------------------------------------------------------
- Sentiment labels are derived from star ratings, not human annotations.
- Authenticity labels are reviewer-associated, not verified per-review fraud.
- Probabilities are model scores, not calibrated confidence values.
- A low-confidence prediction is marked 'uncertain' rather than forced.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _ROOT / "models"

SENTIMENT_PATH     = MODELS_DIR / "sentiment_pipeline.joblib"
AUTHENTICITY_PATH  = MODELS_DIR / "authenticity_pipeline.joblib"

# A prediction is marked 'uncertain' when the winning class probability
# is below this threshold.
UNCERTAINTY_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class SinglePrediction:
    """Result for one classifier."""
    label: str                        # Predicted class label, or 'uncertain'
    probabilities: dict[str, float]   # {class_name: probability}
    is_uncertain: bool                # True when max probability < threshold
    model_available: bool             # False when the pipeline file is missing
    caveat: str                       # Fixed disclaimer about label meaning
    error: Optional[str] = None       # Set if something went wrong at runtime


@dataclass
class PredictionResult:
    """Combined output of both classifiers for a single input text."""
    text_preview: str                 # First 120 chars of input, for traceability
    sentiment: SinglePrediction
    authenticity: SinglePrediction


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

class GhostSignalClassifier:
    """
    Wraps the two trained sklearn pipelines and provides a unified
    prediction interface.

    Both pipelines are loaded lazily on first use and cached for the
    lifetime of the instance.  If a pipeline file is missing, that
    classifier returns a graceful 'unavailable' result rather than
    raising an exception.
    """

    _SENTIMENT_CAVEAT = (
        "Sentiment is derived from star ratings (1-2 → Negative, 4-5 → Positive). "
        "This model supports only two classes: Positive and Negative. "
        "It has no Neutral class — reviews with moderate or mixed language are "
        "forced into one of these two categories and may be misclassified. "
        "The underlying TF-IDF with unigram and bigram features may not reliably "
        "capture negation, sarcasm, or relationships between clauses "
        "(for example, 'not bad' can be partly overridden by the strong individual "
        "negative signals of the words 'not' and 'bad'). "
        "Probabilities are model scores, not calibrated confidence values."
    )
    _AUTHENTICITY_CAVEAT = (
        "Authenticity labels are reviewer-associated proxy labels, "
        "not verified per-review fraud determinations. "
        "FAKE means the reviewer account was labeled fake in the Amazon source dataset; "
        "HONEST means the reviewer account was labeled honest. "
        "This label is propagated from the reviewer profile to all their reviews — "
        "it does not reflect a judgment about the individual review submitted here. "
        "This model was not specifically trained to detect promotional or marketing "
        "language. Text written in an enthusiastic or promotional style may receive "
        "a high HONEST score because such language overlaps with writing patterns "
        "of reviewers labeled honest in the training data. "
        "A prediction of HONEST is not proof that this review is genuine. "
        "A prediction of FAKE is not proof that this review is fraudulent. "
        "Probabilities are model scores, not calibrated confidence values."
    )

    def __init__(self) -> None:
        self._sentiment_pipeline = None
        self._authenticity_pipeline = None
        self._sentiment_loaded: bool | None = None
        self._authenticity_loaded: bool | None = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_sentiment(self) -> bool:
        if self._sentiment_loaded is not None:
            return self._sentiment_loaded
        try:
            import joblib
            self._sentiment_pipeline = joblib.load(SENTIMENT_PATH)
            self._sentiment_loaded = True
            log.info("Sentiment pipeline loaded from %s", SENTIMENT_PATH)
        except FileNotFoundError:
            log.warning("Sentiment pipeline not found at %s. Run train.py first.", SENTIMENT_PATH)
            self._sentiment_loaded = False
        except Exception as exc:
            log.error("Failed to load sentiment pipeline: %s", exc)
            self._sentiment_loaded = False
        return self._sentiment_loaded

    def _load_authenticity(self) -> bool:
        if self._authenticity_loaded is not None:
            return self._authenticity_loaded
        try:
            import joblib
            self._authenticity_pipeline = joblib.load(AUTHENTICITY_PATH)
            self._authenticity_loaded = True
            log.info("Authenticity pipeline loaded from %s", AUTHENTICITY_PATH)
        except FileNotFoundError:
            log.warning(
                "Authenticity pipeline not found at %s. Run train.py first.", AUTHENTICITY_PATH
            )
            self._authenticity_loaded = False
        except Exception as exc:
            log.error("Failed to load authenticity pipeline: %s", exc)
            self._authenticity_loaded = False
        return self._authenticity_loaded

    # ------------------------------------------------------------------
    # Internal prediction helper
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        pipeline,
        text: str,
        caveat: str,
        model_available: bool = True,
    ) -> SinglePrediction:
        """
        Run one pipeline on a single text string.
        Returns a SinglePrediction with label, probabilities, and flags.
        """
        if not model_available or pipeline is None:
            return SinglePrediction(
                label="unavailable",
                probabilities={},
                is_uncertain=True,
                model_available=False,
                caveat=caveat,
                error="Model not loaded. Run train.py to generate model artifacts.",
            )

        if not text or not text.strip():
            return SinglePrediction(
                label="unavailable",
                probabilities={},
                is_uncertain=True,
                model_available=True,
                caveat=caveat,
                error="Empty input text. Cannot classify.",
            )

        try:
            proba_array = pipeline.predict_proba([text])[0]
            classes = [str(c) for c in pipeline.classes_]
            proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba_array)}

            max_prob = max(proba_array)
            predicted_label = str(classes[int(proba_array.argmax())])
            is_uncertain = float(max_prob) < UNCERTAINTY_THRESHOLD

            if is_uncertain:
                display_label = "uncertain"
            else:
                display_label = predicted_label

            return SinglePrediction(
                label=display_label,
                probabilities=proba_dict,
                is_uncertain=is_uncertain,
                model_available=True,
                caveat=caveat,
            )
        except Exception as exc:
            log.error("Prediction error: %s", exc)
            return SinglePrediction(
                label="unavailable",
                probabilities={},
                is_uncertain=True,
                model_available=True,
                caveat=caveat,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> PredictionResult:
        """
        Classify a single review text with both classifiers.

        Parameters
        ----------
        text : Raw review text string.

        Returns
        -------
        PredictionResult
            .sentiment      — SinglePrediction (Positive / Negative / uncertain)
            .authenticity   — SinglePrediction (FAKE / HONEST / uncertain)

        Each SinglePrediction contains:
            .label          — predicted class, or 'uncertain', or 'unavailable'
            .probabilities  — {class_name: probability score}
            .is_uncertain   — True when max probability < 0.60
            .model_available — False if the pipeline file was not found
            .caveat         — fixed disclaimer about what the label means
            .error          — error message if something failed

        Notes
        -----
        - Probabilities are model scores, not calibrated confidence values.
        - 'uncertain' does not mean the content is safe or authentic.
        - 'FAKE' does not mean the individual review is proven fraudulent.
        """
        sent_ok  = self._load_sentiment()
        auth_ok  = self._load_authenticity()

        sentiment_pred = self._run_pipeline(
            self._sentiment_pipeline,
            text,
            caveat=self._SENTIMENT_CAVEAT,
            model_available=sent_ok,
        )
        authenticity_pred = self._run_pipeline(
            self._authenticity_pipeline,
            text,
            caveat=self._AUTHENTICITY_CAVEAT,
            model_available=auth_ok,
        )

        return PredictionResult(
            text_preview=(text[:120] + "...") if len(text) > 120 else text,
            sentiment=sentiment_pred,
            authenticity=authenticity_pred,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """
        Classify a list of review texts.  Calls predict() per item.

        For large batches consider calling the internal pipelines directly;
        this method prioritises clarity over throughput.
        """
        return [self.predict(t) for t in texts]

    def is_ready(self) -> dict[str, bool]:
        """Return which pipelines are loaded and ready."""
        return {
            "sentiment": bool(self._load_sentiment()),
            "authenticity": bool(self._load_authenticity()),
        }

    # ------------------------------------------------------------------
    # Convenience repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s = self._sentiment_loaded
        a = self._authenticity_loaded
        return (
            f"GhostSignalClassifier("
            f"sentiment={'loaded' if s else 'not loaded'}, "
            f"authenticity={'loaded' if a else 'not loaded'})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

# A shared default instance. Import and use directly for simple cases:
#   from src.model.classifier import default_classifier
#   result = default_classifier.predict("some review text")
default_classifier = GhostSignalClassifier()


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json

    SAMPLES = [
        "This product is absolutely amazing! Works perfectly, would buy again.",
        "Complete garbage. Stopped working after two days. Waste of money.",
        "It's okay. Nothing special but does the job.",
        "Best purchase I ever made. Five stars without hesitation!",
        "Do not buy this. Fake reviews everywhere. Product is a scam.",
    ]

    clf = GhostSignalClassifier()
    print("Model status:", clf.is_ready())
    print()

    for sample in SAMPLES:
        result = clf.predict(sample)
        print(f"Text    : {result.text_preview}")
        print(f"Sentiment     → {result.sentiment.label}  "
              f"(probs: {result.sentiment.probabilities}, "
              f"uncertain: {result.sentiment.is_uncertain})")
        print(f"Authenticity  → {result.authenticity.label}  "
              f"(probs: {result.authenticity.probabilities}, "
              f"uncertain: {result.authenticity.is_uncertain})")
        if result.sentiment.error:
            print(f"  Sentiment error: {result.sentiment.error}")
        if result.authenticity.error:
            print(f"  Authenticity error: {result.authenticity.error}")
        print()
