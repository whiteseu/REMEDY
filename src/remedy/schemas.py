"""Validate model records before any field affects routing or final scores."""

from .state import probability

VIEWS = {"image", "clinical_context"}


def _object(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _text(value):
    if not isinstance(value, str):
        raise ValueError("Expected a string")
    return value


def _list(value):
    if not isinstance(value, list):
        raise ValueError("Expected an array")
    return value


def _flag(value):
    if not isinstance(value, bool):
        raise ValueError("Expected a boolean")
    return value


def _strings(value):
    return [_text(x) for x in _list(value)]


def _label(value, labels):
    if value not in labels:
        raise ValueError("Record names a label outside the closed vocabulary")
    return value


def validate_plan(raw, labels):
    r = _object(raw)
    complexity = r["case_complexity"]
    if complexity not in {"low", "medium", "high"}:
        raise ValueError("Invalid case complexity")
    priority = _strings(r["view_priority"])
    if set(priority) - VIEWS or len(set(priority)) != len(priority):
        raise ValueError("Invalid view priority")
    candidates = _strings(r["candidates"])
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate planner candidates")
    # Out-of-vocabulary or ungrounded hypotheses are filtered by deterministic admission.
    unresolved = []
    for item in _list(r["unresolved"]):
        item = _object(item)
        unresolved.append({"hypothesis": _text(item["hypothesis"]),
                           "evidence_needed": _text(item["evidence_needed"])})
    return {"case_complexity": complexity, "view_priority": priority,
            "candidates": candidates, "unresolved": unresolved,
            "request_counterfactual": _flag(r["request_counterfactual"]),
            "rationale": _text(r["rationale"])}


def _strengths(items, labels):
    result = []
    for item in _list(items):
        item = _object(item)
        anchor = _text(item["anchor"]).strip()
        if not anchor:
            raise ValueError("Evidence strength requires an anchor")
        result.append({"label": _label(item["label"], labels),
                       "strength": probability(item["strength"]), "anchor": anchor})
    return result


def validate_tool(stage, raw, labels, candidates):
    r = _object(raw)
    usable = _flag(r["usable"])
    if not usable:
        return {"usable": False}
    out = {"usable": True}
    if stage in {"T2", "T3"}:
        findings = []
        for item in _list(r["findings"]):
            item = _object(item)
            anchor = _text(item["anchor"]).strip()
            if not anchor:
                raise ValueError("Every finding requires a query-side anchor")
            findings.append({"finding": _text(item["finding"]), "anchor": anchor})
        out.update(findings=findings, supporting_labels=_strengths(r["supporting_labels"], labels),
                   opposing_labels=_strengths(r["opposing_labels"], labels))
    elif stage == "T4":
        reliable = r["more_reliable_view"]
        if reliable not in VIEWS | {"tie", "none"}:
            raise ValueError("Invalid reliable view")
        out.update(conflict_score=probability(r["conflict_score"]),
                   conflicts=_strings(r["conflicts"]), more_reliable_view=reliable)
    elif stage == "T5":
        for key in ("label_support", "label_distribution"):
            vals = _object(r[key])
            out[key] = {_label(c, labels): probability(v) for c, v in vals.items()}
    elif stage == "T6":
        ranking = []
        for item in _list(r["ranking"]):
            item = _object(item)
            ranking.append({"label": _label(item["label"], candidates),
                            "score": probability(item["score"]), "anchor": _text(item["anchor"])})
        if len({x["label"] for x in ranking}) != len(ranking):
            raise ValueError("Duplicate differential ranking labels")
        out.update(ranking=ranking, pairwise=_strings(r["pairwise"]))
    elif stage == "T7":
        items = []
        for item in _list(r["per_candidate"]):
            item = _object(item)
            verdict = item["verdict"]
            if verdict not in {"retain", "weaken", "undetermined", "rule_out"}:
                raise ValueError("Invalid counterfactual verdict")
            items.append({"label": _label(item["label"], candidates),
                          "expected_sign": _text(item["expected_sign"]),
                          "observation": _text(item["observation"]), "verdict": verdict,
                          "strength": probability(item["strength"]),
                          "anchor": _text(item["anchor"])})
        if len({x["label"] for x in items}) != len(items):
            raise ValueError("Duplicate counterfactual labels")
        out["per_candidate"] = items
    else:
        raise ValueError("Unknown evidence stage")
    return out


def validate_scores(raw, labels):
    r = _object(raw)
    scores = {}
    for item in _list(r["ranked_candidates"]):
        item = _object(item)
        label = _label(item["label"], labels)
        if label in scores:
            raise ValueError("Duplicate score coordinate")
        scores[label] = probability(item["score"], "score")
    if set(scores) != set(labels):
        raise ValueError("The score vector must cover every direct label")
    return {"ranked_candidates": [{"label": c, "score": scores[c]} for c in labels]}


def validate_ecc(raw, labels, usable_stages):
    out = validate_scores(raw, labels)
    out["confidence"] = probability(raw["confidence"], "confidence")
    for key in ("evidence_chain", "counter_evidence"):
        anchors = []
        for item in _list(raw[key]):
            item = _object(item)
            anchor = _text(item["anchor"]).strip()
            if item["tool"] not in usable_stages or not anchor:
                raise ValueError("ECC must cite an executed usable tool and a nonempty anchor")
            anchors.append({"label": _label(item["label"], labels),
                            "tool": item["tool"], "anchor": anchor})
        out[key] = anchors
    anchored = {item["label"] for key in ("evidence_chain", "counter_evidence") for item in out[key]}
    if anchored != set(labels):
        raise ValueError("ECC must name an evidence record behind every direct-label score")
    return out


SCHEMAS = {
    "RCMP": {"case_complexity": "low|medium|high", "view_priority": ["image", "clinical_context"],
             "candidates": ["label"], "unresolved": [{"hypothesis": "label", "evidence_needed": "text"}],
             "request_counterfactual": "boolean", "rationale": "text"},
    "T2": {"usable": "boolean", "findings": [{"finding": "text", "anchor": "location/waveform"}],
           "supporting_labels": [{"label": "label", "strength": "[0,1]", "anchor": "text"}],
           "opposing_labels": [{"label": "label", "strength": "[0,1]", "anchor": "text"}]},
    "T3": {"usable": "boolean", "findings": [{"finding": "text", "anchor": "quoted text/value"}],
           "supporting_labels": [{"label": "label", "strength": "[0,1]", "anchor": "text"}],
           "opposing_labels": [{"label": "label", "strength": "[0,1]", "anchor": "text"}]},
    "T4": {"usable": "boolean", "conflict_score": "[0,1]", "conflicts": ["text"],
           "more_reliable_view": "image|clinical_context|tie|none"},
    "T5": {"usable": "boolean", "label_support": {"label": "[0,1]"},
           "label_distribution": {"label": "[0,1]"}},
    "T6": {"usable": "boolean", "ranking": [{"label": "candidate", "score": "[0,1]", "anchor": "text"}],
           "pairwise": ["pairwise verdict with discriminating evidence"]},
    "T7": {"usable": "boolean", "per_candidate": [{"label": "candidate", "expected_sign": "text",
            "observation": "text", "verdict": "retain|weaken|undetermined|rule_out",
            "strength": "[0,1]", "anchor": "text"}]},
    "ECC": {"ranked_candidates": [{"label": "every direct label, exactly once", "score": "[0,1]"}],
            "evidence_chain": [{"label": "label", "tool": "executed usable tool", "anchor": "text"}],
            "counter_evidence": [{"label": "label", "tool": "executed usable tool", "anchor": "text"}],
            "confidence": "[0,1]"},
    "READOUT": {"ranked_candidates": [{"label": "every direct label, exactly once", "score": "[0,1]"}]},
}
