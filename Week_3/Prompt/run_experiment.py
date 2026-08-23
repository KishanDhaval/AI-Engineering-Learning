from reviews import reviews
from ground_truth import ground_truth
from prompts import PROMPT_VARIANTS
from analyzer import analyze
from evaluator import judge


def run():
    results = []

    for variant_name in PROMPT_VARIANTS:
        print(f"Evaluating prompt variant: {variant_name}...", flush=True)
        sentiment_correct = 0
        issues_scores = []
        summary_scores = []
        parse_failures = 0

        for review, gold in zip(reviews, ground_truth):
            pred = analyze(review, variant_name)

            if not pred.get("ok"):
                parse_failures += 1
                continue

            if pred["sentiment"] == gold["sentiment"]:
                sentiment_correct += 1

            scores = judge(
                review,
                gold["key_issues"], pred["key_issues"],
                gold["summary"], pred["summary"],
            )
            if scores.get("issues_score") is not None:
                issues_scores.append(scores["issues_score"])
            if scores.get("summary_score") is not None:
                summary_scores.append(scores["summary_score"])

        n = len(reviews)
        results.append({
            "variant": variant_name,
            "sentiment_acc": round(sentiment_correct / n, 2),
            "issues_avg": round(sum(issues_scores) / len(issues_scores), 2) if issues_scores else None,
            "summary_avg": round(sum(summary_scores) / len(summary_scores), 2) if summary_scores else None,
            "parse_failures": parse_failures,
        })

    print(f"{'Variant':<18}{'Sentiment Acc':<16}{'Issues (1-5)':<15}{'Summary (1-5)':<15}{'Parse Fails'}")
    print("-" * 75)
    for r in results:
        issues = r['issues_avg'] or "N/A"
        summary = r['summary_avg'] or "N/A"
        print(f"{r['variant']:<18}{r['sentiment_acc']:<16}{issues:<15}{summary:<15}{r['parse_failures']}")

run()