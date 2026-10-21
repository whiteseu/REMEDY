"""Deterministic CKM, DDTR and leave-one-label-out RSF arithmetic."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


def probability(value, name="probability") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class Neighbor:
    similarity: float
    labels: tuple[str, ...]

    def __post_init__(self):
        if isinstance(self.similarity, bool) or not isinstance(self.similarity, (int, float)) or not math.isfinite(self.similarity) or not -1 <= self.similarity <= 1:
            raise ValueError("Neighbor similarity must be a finite cosine in [-1, 1]")
        if len(set(self.labels)) != len(self.labels) or any(not isinstance(c, str) for c in self.labels):
            raise ValueError("Neighbor labels must be unique strings")


@dataclass(frozen=True)
class RetrievalState:
    weights: tuple[float, ...]
    support: dict[str, float]
    votes: dict[str, float]
    raised: tuple[str, ...]
    grounding: dict[str, dict]


def build_state(neighbors: Sequence[Neighbor], labels: Sequence[str], knowledge: Mapping[str, dict]) -> RetrievalState:
    if not neighbors:
        raise ValueError("A nonempty retrieved neighborhood is required")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Direct labels must be nonempty and unique")
    unknown = set().union(*(set(n.labels) for n in neighbors)) - set(labels)
    if unknown:
        raise ValueError("Map neighbor labels to the direct label space before inference")
    affinities = [max(n.similarity, 0.0) for n in neighbors]
    total = sum(affinities)
    weights = tuple(a / total for a in affinities) if total else (1 / len(neighbors),) * len(neighbors)
    r = {c: sum(w for n, w in zip(neighbors, weights) if c in n.labels) for c in labels}
    q = {c: sum(c in n.labels for n in neighbors) / len(neighbors) for c in labels}
    raised = tuple(c for c in labels if q[c] > 0)
    absent = set(raised) - knowledge.keys()
    if absent:
        raise ValueError("Knowledge entries are required for every retrieval-raised label")
    return RetrievalState(weights, r, q, raised, {c: knowledge[c] for c in raised})


def disagreement(votes: Mapping[str, float], raised: Sequence[str]) -> float:
    """Mean 4*q*(1-q) over the raised coordinates, zero for the empty set."""
    values = [probability(votes[c], "vote") for c in raised]
    return sum(4 * q * (1 - q) for q in values) / len(values) if values else 0.0


def leave_one_out_disagreement(state: RetrievalState, label: str) -> float:
    if label not in state.votes:
        raise ValueError("Unknown direct label")
    return disagreement(state.votes, [c for c in state.raised if c != label])


def admissible_labels(state: RetrievalState, labels: Sequence[str]) -> tuple[str, ...]:
    differentials = {d for entry in state.grounding.values() for d in entry.get("differentials", [])}
    allowed = set(state.raised) | (differentials & set(labels))
    return tuple(c for c in labels if c in allowed)


def routing_gate(vote_disagreement, cross_conflict, missing, complexity, config) -> dict:
    dv = probability(vote_disagreement, "vote disagreement")
    dc = probability(cross_conflict, "cross-view conflict")
    sources = {
        "state": dv >= config.tau_vote,
        "evidence": dc >= config.tau_cross or bool(missing),
        "plan": complexity == "high",
    }
    return {**sources, "open": any(sources.values())}


def fuse(scores: Mapping[str, float], state: RetrievalState, config) -> dict:
    """alpha_c = alpha0 * (1 - mean disagreement on raised labels OTHER than c)."""
    if set(scores) != set(config.labels):
        raise ValueError("ECC must score every direct label exactly once")
    alpha, final = {}, {}
    for c in config.labels:
        s = probability(scores[c], "ECC score")
        r = probability(state.support[c], "retrieval support")
        alpha[c] = config.alpha0 * (1 - leave_one_out_disagreement(state, c))
        final[c] = (1 - alpha[c]) * s + alpha[c] * r
    return {"alpha": alpha, "scores": final,
            "predictions": [c for c in config.labels if final[c] >= config.delta]}


def derived_no_mapped_finding(scores: Mapping[str, float], delta=0.5) -> tuple[float, bool]:
    """Optional BRSET derived coordinate; do not include it in direct-coordinate RSF."""
    if not scores:
        raise ValueError("At least one direct pathology score is required")
    peak = max(probability(v) for v in scores.values())
    return 1 - peak, peak < probability(delta, "threshold")
