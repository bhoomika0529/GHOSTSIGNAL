"""
GHOSTSIGNAL — Review Analyzer
Streamlit application entry point.

Launch:
    streamlit run app.py

Must be run from the project root so that the src/ package is on sys.path.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `src.*` imports resolve
# correctly regardless of the directory Streamlit is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.services.investigation_service import analyze_review, get_model_status

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GHOSTSIGNAL — Review Analyzer",
    page_icon="👻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal polish without external dependencies
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .gs-label-positive  { color: #1f8f1f; font-weight: 700; font-size: 1.3rem; }
    .gs-label-negative  { color: #c0392b; font-weight: 700; font-size: 1.3rem; }
    .gs-label-fake      { color: #c0392b; font-weight: 700; font-size: 1.3rem; }
    .gs-label-honest    { color: #1f8f1f; font-weight: 700; font-size: 1.3rem; }
    .gs-label-uncertain { color: #b7950b; font-weight: 700; font-size: 1.3rem; }
    .gs-label-unavail   { color: #7f8c8d; font-weight: 700; font-size: 1.3rem; }
    .gs-prob-bar-wrap   { background: #ecf0f1; border-radius: 6px; height: 14px;
                          margin: 4px 0 10px 0; overflow: hidden; }
    .gs-prob-bar        { height: 14px; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper — render one probability bar row
# ---------------------------------------------------------------------------
def _render_prob_row(class_name: str, prob: float, color: str) -> None:
    pct = round(prob * 100, 1)
    col_label, col_bar, col_pct = st.columns([2, 5, 1])
    with col_label:
        st.markdown(f"**{class_name}**")
    with col_bar:
        st.markdown(
            f'<div class="gs-prob-bar-wrap">'
            f'<div class="gs-prob-bar" style="width:{pct}%;background:{color};"></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_pct:
        st.markdown(f"{pct}%")


# ---------------------------------------------------------------------------
# Helper — render a SinglePrediction result card
# ---------------------------------------------------------------------------
def _render_prediction_card(
    heading: str,
    pred: dict,
    label_map: dict[str, str],   # label value → display name
    color_map: dict[str, str],   # label value → hex color
    css_class_map: dict[str, str],
) -> None:
    """Render one classifier result (sentiment or authenticity) as a Streamlit block."""

    label = pred.get("label", "unavailable")
    is_uncertain = pred.get("is_uncertain", True)
    model_available = pred.get("model_available", False)
    probabilities = pred.get("probabilities") or {}
    error_msg = pred.get("error")
    caveat = pred.get("caveat", "")

    st.subheader(heading)

    # --- Model unavailable ---
    if not model_available:
        st.markdown(
            '<span class="gs-label-unavail">⚠ Model unavailable</span>',
            unsafe_allow_html=True,
        )
        if error_msg:
            st.caption(error_msg)
        return

    # --- Empty / runtime error ---
    if label == "unavailable":
        st.markdown(
            '<span class="gs-label-unavail">⚠ Unavailable</span>',
            unsafe_allow_html=True,
        )
        if error_msg:
            st.caption(error_msg)
        return

    # --- Uncertain ---
    if label == "uncertain" or is_uncertain:
        st.markdown(
            f'<span class="gs-label-uncertain">~ Uncertain</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "The highest predicted probability is below 60 %. "
            "The probability scores below indicate the direction but are not reliable enough for a firm prediction."
        )
    else:
        display_name = label_map.get(label, label.capitalize())
        css_cls = css_class_map.get(label, "gs-label-unavail")
        icon = "✅" if label in ("Positive", "HONEST") else "❌"
        st.markdown(
            f'<span class="{css_cls}">{icon} {display_name}</span>',
            unsafe_allow_html=True,
        )

    # --- Probability bars ---
    if probabilities:
        st.markdown("**Probability scores**")
        for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
            bar_color = color_map.get(cls, "#95a5a6")
            _render_prob_row(cls, prob, bar_color)


# ---------------------------------------------------------------------------
# Main — model status banner
# ---------------------------------------------------------------------------
def _render_status_banner() -> None:
    try:
        status = get_model_status()
    except Exception as exc:
        st.warning(
            "⚠ Model status could not be determined. "
            "The application may not function correctly. "
            f"Detail: {type(exc).__name__}.",
            icon="⚠️",
        )
        return
    if not status.get("sentiment") or not status.get("authenticity"):
        missing = [k for k, v in status.items() if not v]
        st.warning(
            f"⚠ The following model(s) could not be loaded: **{', '.join(missing)}**. "
            "Run `python -m src.model.train` from the project root to generate the model artifacts.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:

    # --- Header ---
    st.title("👻 GHOSTSIGNAL")
    st.markdown(
        "**Review Analyzer** — paste any product or business review below to "
        "receive independent sentiment and authenticity estimates from the trained models."
    )
    st.divider()

    # --- Model status ---
    _render_status_banner()

    # --- Input ---
    review_text = st.text_area(
        label="Review text",
        placeholder=(
            "Paste a product review, service review, or any review text here…\n\n"
            "Example: \"Absolutely love this product! Works perfectly, would buy again.\""
        ),
        height=160,
        key="review_input",
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("🔍 Analyze Review", type="primary", use_container_width=True)

    # --- Trigger analysis and store result in session state ---
    if analyze_clicked:
        if not review_text or not review_text.strip():
            # Clear any stale result so old predictions are not shown below
            # the validation error from a previous successful analysis.
            st.session_state.pop("gs_result", None)
            st.error("Please enter some review text before clicking Analyze.", icon="🚫")
        else:
            with st.spinner("Analyzing…"):
                st.session_state["gs_result"] = analyze_review(review_text)

    # --- Render stored result (persists across Streamlit reruns) ---
    if "gs_result" in st.session_state:
        result = st.session_state["gs_result"]

        st.divider()

        # Service-layer validation failure
        if not result.get("ok"):
            st.error(result.get("error_message", "An unknown error occurred."), icon="🚫")
            return

        # Text preview
        preview = result.get("text_preview", "")
        if preview:
            st.caption(f'Analyzed: "{preview}"')

        # Short-text warning — displayed when input has fewer than 5 words.
        # The prediction and probabilities are unchanged; this note simply
        # alerts the user that very few model features were active.
        if result.get("is_short_text"):
            st.warning(
                "⚠ **Short input detected.** This review contains fewer than 5 words. "
                "Very short text activates very few model features, which can produce "
                "extreme-looking probability scores. "
                "The prediction may not reflect genuine model certainty.",
                icon="⚠️",
            )

        # --- Results columns ---
        col_sent, col_auth = st.columns(2, gap="large")

        with col_sent:
            pred = result.get("sentiment") or {}
            _render_prediction_card(
                heading="📊 Sentiment",
                pred=pred,
                label_map={"Positive": "Positive", "Negative": "Negative"},
                color_map={"Positive": "#27ae60", "Negative": "#e74c3c"},
                css_class_map={
                    "Positive": "gs-label-positive",
                    "Negative": "gs-label-negative",
                },
            )
            if pred.get("caveat"):
                with st.expander("ℹ What does this mean?", expanded=False):
                    st.caption(pred["caveat"])
                    st.caption(
                        "⚠ **Binary classification only.** "
                        "This model was trained on 1–2 star reviews (Negative) and "
                        "4–5 star reviews (Positive). "
                        "It has no Neutral class. "
                        "Reviews with moderate or mixed language are forced into one "
                        "of these two categories."
                    )

        with col_auth:
            pred = result.get("authenticity") or {}
            _render_prediction_card(
                heading="🔎 Authenticity",
                pred=pred,
                label_map={"FAKE": "Likely Fake", "HONEST": "Likely Honest"},
                color_map={"FAKE": "#e74c3c", "HONEST": "#27ae60"},
                css_class_map={
                    "FAKE": "gs-label-fake",
                    "HONEST": "gs-label-honest",
                },
            )
            if pred.get("caveat"):
                with st.expander("ℹ What does this mean?", expanded=False):
                    st.caption(pred["caveat"])
                    st.caption(
                        "⚠ **Important:** "
                        "These labels are reviewer-associated proxy labels derived from "
                        "Amazon reviewer account metadata — they are not a verified "
                        "determination that this specific review is fraudulent or genuine. "
                        "This model was not specifically trained to detect promotional or "
                        "marketing language. "
                        "Enthusiastic or promotional wording may receive a high HONEST "
                        "score because it overlaps with the writing patterns of reviewers "
                        "labeled honest in the training data. "
                        "Do not use this prediction as proof of any claim."
                    )

        # --- Disclaimer footer ---
        st.divider()
        st.caption(
            "GHOSTSIGNAL uses trained machine-learning models to estimate sentiment "
            "and reviewer-associated authenticity patterns. Predictions are probabilistic "
            "and may be incorrect. They are intended for investigative context only and "
            "should not be treated as proof of any claim."
        )

    # --- First-load hint ---
    elif not analyze_clicked:
        st.info(
            "Enter a review above and click **Analyze Review** to see predictions.",
            icon="💡",
        )


if __name__ == "__main__":
    main()
