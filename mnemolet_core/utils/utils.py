from typing import Any


def filter_by_min_score(
    raw: list[dict[str, Any]], min_score: float
) -> list[dict[str, Any]]:
    """
    Return only results with score >= min_score.
    """
    filtered = []
    for r in raw:
        try:
            score = float(r.get("score", 0))
            if score >= min_score:
                filtered.append(r)
        except (ValueError, TypeError):
            continue
    return filtered
