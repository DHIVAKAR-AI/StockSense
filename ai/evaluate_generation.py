"""
ai/evaluate_generation.py
--------------------------
Task-appropriate evaluation for the natural-language generation features
(assistant replies, explanations, insight summaries). Since these are not
classification outputs, accuracy/precision/recall don't apply — instead
we check:

  * indicator-keyword coverage  -> does the explanation actually mention
                                    the technical signals behind it?
  * explanation length          -> is it in a sensible, readable range?
  * distinct-2 lexical diversity -> do repeated generations avoid being
                                    identical / robotic?
"""

import re
from collections import Counter

from ai.insight_generator import build_explanation, build_insight_summary, parse_query
from ml.predictor import list_symbols, predict_trend

FEATURE_KEYWORD_HINTS = {
    "rsi14": ["rsi", "momentum"],
    "daily_return": ["momentum", "return"],
    "sma5": ["moving-average", "moving average", "crossover"],
    "sma10": ["moving-average", "moving average", "crossover"],
    "volatility10": ["volatility", "stable", "swings"],
}


def keyword_coverage(explanation: str, top_features: list[str]) -> float:
    text = explanation.lower()
    hits = 0
    for feature in top_features:
        hints = FEATURE_KEYWORD_HINTS.get(feature, [feature.replace("_", " ")])
        if any(h in text for h in hints):
            hits += 1
    return hits / len(top_features) if top_features else 0.0


def distinct_2(texts: list[str]) -> float:
    bigrams = Counter()
    total = 0
    for t in texts:
        tokens = re.findall(r"[a-zA-Z']+", t.lower())
        pairs = list(zip(tokens, tokens[1:]))
        bigrams.update(pairs)
        total += len(pairs)
    return len(bigrams) / total if total else 0.0


def run_evaluation():
    print("=" * 70)
    print("STOCKSENSE GENERATIVE AI FEATURES - EVALUATION")
    print("=" * 70)

    symbols = [s["symbol"] for s in list_symbols()]

    # --- Assistant / explanation evaluation ---
    coverages, lengths = [], []
    sample_queries = [
        "Should I invest in {s} right now?",
        "What's the risk on {s}?",
        "Give me a 3-day trend check on {s}",
        "Is {s} looking bullish?",
    ]
    tested = 0
    for symbol in symbols:
        prediction = predict_trend(symbol)
        top_features = sorted(
            prediction["feature_importances"], key=prediction["feature_importances"].get, reverse=True
        )[:3]
        for template in sample_queries:
            message = template.format(s=symbol)
            parsed = parse_query(message)
            explanation = build_explanation(prediction, intent=parsed.intent)
            coverages.append(keyword_coverage(explanation, top_features))
            lengths.append(len(explanation.split()))
            tested += 1

    avg_coverage = sum(coverages) / len(coverages)
    avg_length = sum(lengths) / len(lengths)

    print(f"\nAssistant responses tested: {tested} queries across {len(symbols)} stocks")
    print(f"Average indicator-keyword coverage: {avg_coverage:.2f}")
    print(f"Average explanation length: {avg_length:.0f} words (target range: 25-60)")

    # --- Insight summary diversity evaluation ---
    repeats_per_stock = 10
    sample_stocks = symbols[:5]
    all_summaries = []
    for symbol in sample_stocks:
        prediction = predict_trend(symbol)
        summaries = [build_insight_summary(prediction) for _ in range(repeats_per_stock)]
        all_summaries.extend(summaries)

    diversity = distinct_2(all_summaries)
    duplicates = len(all_summaries) - len(set(all_summaries))

    print(f"\nInsight summaries generated per stock: {repeats_per_stock} repeats x {len(sample_stocks)} stocks")
    print(f"Distinct-2 (lexical diversity) score: {diversity:.2f}")
    print(f"Duplicate summaries produced: {duplicates} / {len(all_summaries)}")

    print("\nEvaluation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
