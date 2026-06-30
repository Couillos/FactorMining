def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard_stability(formula_sets: list[set[str]]) -> float:
    if len(formula_sets) < 2:
        return 0.0
    pairs = 0
    total = 0.0
    for i in range(len(formula_sets)):
        for j in range(i + 1, len(formula_sets)):
            total += _jaccard(formula_sets[i], formula_sets[j])
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def jaccard_pass(formula_sets: list[set[str]], threshold: float = 0.7) -> bool:
    return jaccard_stability(formula_sets) >= threshold
