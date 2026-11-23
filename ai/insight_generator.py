"""
ai/insight_generator.py
------------------------
Offline, template-based "generative" layer that sits on top of the core
Random Forest model. It:

  1. Parses free-text investment queries (parse_query)
  2. Explains a prediction in plain language (build_explanation)
  3. Produces a friendly narrative insight summary (build_insight_summary)

Every function here takes and returns plain dicts/strings, deliberately
matching the shape a real LLM call (OpenAI / local Ollama) would use —
see USE_LLM below. That means this module can be swapped for a live LLM
call without changing any of the calling code in backend/main.py.
"""

import random
import re
from dataclasses import dataclass

from ml.predictor import list_symbols

USE_LLM = False  # flip to True and implement _call_llm() to use a real model

INTENT_KEYWORDS = {
    "risk_check": ["risk", "risky", "volatile", "volatility", "safe"],
    "compare": ["compare", "vs", "versus", "better than"],
    "trend_check": ["invest", "buy", "trend", "should i", "outlook", "forecast"],
}

DAY_PATTERN = re.compile(r"(\d+)\s*-?\s*day")


@dataclass
class ParsedQuery:
    symbol: str
    intent: str
    horizon_days: int
    raw_message: str


def _known_symbols() -> list[str]:
    return [s["symbol"] for s in list_symbols()]


def parse_query(message: str) -> ParsedQuery:
    """Very small, dependency-free NLU: symbol lookup + keyword intent."""
    text = message.lower()

    symbol = None
    for known in _known_symbols():
        if known.lower() in text or known.lower() in text.replace(" ", ""):
            symbol = known
            break
    if symbol is None:
        # fall back to company-name fragments could go here; default to
        # the first known symbol so the assistant always has something
        # useful to say instead of failing hard.
        symbol = _known_symbols()[0]

    intent = "trend_check"
    for candidate_intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            intent = candidate_intent
            break

    match = DAY_PATTERN.search(text)
    horizon_days = int(match.group(1)) if match else 5

    return ParsedQuery(symbol=symbol, intent=intent, horizon_days=horizon_days, raw_message=message)


def _reasons_from_prediction(stock_avg_volatility: float, prediction: dict, top_n: int = 3) -> list[str]:
    """
    Grounds the explanation in the model's own top-ranked features (rather
    than an independent set of hand-picked thresholds), so the generated
    text reliably reflects what actually drove the prediction.
    """
    values = prediction["feature_values"]
    importances = prediction["feature_importances"]
    top_features = sorted(importances, key=importances.get, reverse=True)[:top_n]

    reasons = []
    for feature in top_features:
        if feature == "rsi14":
            rsi = values["rsi14"]
            zone = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
            reasons.append(f"an RSI14 of {rsi:.1f} ({zone} territory)")

        elif feature == "daily_return":
            ret = values["daily_return"]
            direction = "positive" if ret > 0 else "negative" if ret < 0 else "flat"
            reasons.append(f"{direction} recent momentum (last daily return {ret * 100:+.2f}%)")

        elif feature in ("sma5", "sma10"):
            if values["sma5"] > values["sma10"]:
                reasons.append("a bullish short-term moving-average crossover (SMA5 above SMA10)")
            else:
                reasons.append("a bearish short-term moving-average crossover (SMA5 below SMA10)")

        elif feature == "volatility10":
            vol = values["volatility10"]
            if vol > stock_avg_volatility * 1.15:
                reasons.append("elevated short-term volatility")
            elif vol < stock_avg_volatility * 0.85:
                reasons.append("subdued, stable short-term volatility")
            else:
                reasons.append("volatility close to the stock's own historical average")

    # de-duplicate while preserving order (sma5/sma10 can both map to the
    # same crossover sentence)
    seen, deduped = set(), []
    for r in reasons:
        if r not in seen:
            deduped.append(r)
            seen.add(r)

    return deduped or ["mixed technical signals with no single dominant driver"]


def build_explanation(prediction: dict, intent: str = "trend_check") -> str:
    """Turns a prediction dict (see ml.predictor.predict_trend) into a
    short, plain-language explanation."""
    if USE_LLM:
        return _call_llm(prediction, intent)  # pragma: no cover

    reasons = _reasons_from_prediction(prediction["avg_volatility"], prediction)
    reason_text = " and ".join(reasons) if len(reasons) <= 2 else ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

    intent_prefix = {
        "risk_check": f"In terms of risk, {prediction['symbol']} currently shows ",
        "compare": f"On its own merits, {prediction['symbol']} currently shows ",
        "trend_check": f"{prediction['symbol']} is currently showing ",
    }.get(intent, f"{prediction['symbol']} is currently showing ")

    return (
        f"{intent_prefix}a {prediction['trend']} signal with "
        f"{prediction['confidence']:.0%} confidence over the next "
        f"{prediction['horizon_days']} trading day(s), driven mainly by {reason_text}. "
        f"This is an automated indication based on historical patterns, not financial "
        f"advice — please do your own research before investing."
    )


OPENERS = [
    "Here's the latest read on {symbol}:",
    "Quick snapshot for {symbol}:",
    "{symbol} at a glance:",
    "Taking a look at {symbol} now:",
]
CLOSERS = [
    "Keep an eye on volume over the next couple of sessions.",
    "Worth revisiting this signal after the next trading day.",
    "As always, treat this as one input among several before deciding.",
    "Consider checking sector-wide trends alongside this signal.",
]


def build_insight_summary(prediction: dict) -> str:
    """Friendly, narrative summary of a prediction — varies its wording
    across calls so repeated generations don't read identically."""
    if USE_LLM:
        return _call_llm_summary(prediction)  # pragma: no cover

    opener = random.choice(OPENERS).format(symbol=prediction["symbol"])
    closer = random.choice(CLOSERS)
    top_feature = max(prediction["feature_importances"].items(), key=lambda kv: kv[1])[0]

    return (
        f"{opener} Predicted Trend: {prediction['trend']}. "
        f"Confidence: {prediction['confidence']:.0%}. "
        f"The strongest signal right now is {top_feature.replace('_', ' ')}. "
        f"{closer}"
    )


# ---------------------------------------------------------------------
# Investment verdict — turns trend + confidence + risk into a concrete
# "is this a good time to invest" style recommendation, e.g.:
#
#   Stock: RELIANCE
#   Predicted Trend: Bullish
#   Confidence: 78%
#   Reason: Positive recent momentum and increasing trading volume.
#   Verdict: 🟢 Good time to consider investing
# ---------------------------------------------------------------------

CONFIDENT_THRESHOLD = 0.55  # above this, the model's signal is trusted more strongly
HIGH_RISK_MULTIPLIER = 1.3  # volatility this far above the stock's own average = elevated risk


def build_verdict(prediction: dict) -> dict:
    """
    Maps (trend, confidence, volatility) onto a plain investment verdict.
    This is a deliberately simple, transparent rule layer on top of the
    model's probabilities — not a second model — so the logic stays easy
    to audit and explain to a non-technical reader.
    """
    trend = prediction["trend"]
    confidence = prediction["confidence"]
    volatility = prediction["feature_values"]["volatility10"]
    avg_volatility = prediction["avg_volatility"] or volatility
    high_risk = volatility > avg_volatility * HIGH_RISK_MULTIPLIER

    if trend == "Bullish" and confidence >= CONFIDENT_THRESHOLD:
        action, emoji, label = "invest", "🟢", "Good time to consider investing"
    elif trend == "Bullish":
        action, emoji, label = "watch", "🟡", "Cautiously positive — momentum is building but confidence is moderate"
    elif trend == "Bearish" and confidence >= CONFIDENT_THRESHOLD:
        action, emoji, label = "avoid", "🔴", "Not a good time to invest right now"
    elif trend == "Bearish":
        action, emoji, label = "caution", "🟡", "Mild downward pressure — proceed with caution"
    else:
        action, emoji, label = "wait", "🟡", "Sideways / uncertain — better to wait for a clearer signal"

    note = None
    if high_risk:
        note = "Volatility is currently elevated for this stock, so treat the signal with extra caution."

    return {
        "action": action,
        "emoji": emoji,
        "label": label,
        "note": note,
        "high_risk": high_risk,
    }


def _call_llm(prediction: dict, intent: str) -> str:  # pragma: no cover
    """Placeholder for a real LLM call (OpenAI / local Ollama).

    Example real implementation:

        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stock market explainer..."},
                {"role": "user", "content": json.dumps(prediction)},
            ],
        )
        return resp.choices[0].message.content
    """
    raise NotImplementedError("Set USE_LLM=False or implement _call_llm().")


def _call_llm_summary(prediction: dict) -> str:  # pragma: no cover
    raise NotImplementedError("Set USE_LLM=False or implement _call_llm_summary().")
