"""Stage projections enforce the manuscript's information-access boundaries."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .config import MethodConfig
from .model import Backend, ModelError, parse_object
from .prompts import system_prompt
from .schemas import validate_ecc, validate_plan, validate_scores, validate_tool
from .state import (Neighbor, admissible_labels, build_state, disagreement, fuse, routing_gate)


@dataclass(frozen=True)
class Case:
    neighbors: tuple[Neighbor, ...]
    image: Path | None = None
    clinical_context: str | dict | None = None
    modality: str = "chest radiograph"


def normalize_knowledge(raw: Mapping, labels) -> dict[str, dict]:
    """Load user-owned grounding; do not include private resources in the code release."""
    if not isinstance(raw, Mapping):
        raise ValueError("Knowledge must map direct labels to entries")
    entries = {}
    for c in labels:
        if c not in raw:
            continue
        entry = raw[c]
        if not isinstance(entry, dict) or not entry.get("criteria"):
            raise ValueError("Each supplied knowledge entry requires diagnostic criteria")
        differentials = entry.get("differentials", [])
        if not isinstance(differentials, list) or any(not isinstance(d, str) for d in differentials):
            raise ValueError("Differential links must be a list of label names")
        entries[c] = {"concepts": entry.get("concepts", entry.get("concept", {})),
                      "criteria": entry["criteria"],
                      "references": entry.get("references", entry.get("standard", [])),
                      "differentials": list(dict.fromkeys(differentials))}
    return entries


class Pipeline:
    """One frozen backend is reused for every model-based stage.

    Per-stage validation allows one repair retry. Normal execution costs six to eight
    model calls with both base views. Fallbacks and retries are counted separately.
    Instances are reusable sequentially; create one instance per concurrent query.
    """

    def __init__(self, backend: Backend, knowledge: Mapping, config: MethodConfig | None = None):
        self.config = config or MethodConfig()
        self.knowledge = normalize_knowledge(knowledge, self.config.labels)
        self.backend = backend
        self.events = []

    def _call(self, stage, payload, validator, image=None):
        for attempt in range(2):
            projection = dict(payload)
            if attempt:
                projection["schema_repair"] = "Previous completion was invalid or unavailable. Return the specified JSON schema."
            try:
                response = self.backend.complete(stage, system_prompt(stage), projection, image=image)
                result = validator(parse_object(response))
                self.events.append({"stage": stage, "attempt": attempt + 1, "status": "ok"})
                return result, False
            except (ValueError, KeyError, TypeError, ModelError) as error:
                self.events.append({"stage": stage, "attempt": attempt + 1,
                                    "status": "failed", "error_type": type(error).__name__})
        return {}, True

    def _tool(self, stage, payload, plan, records, image=None):
        result, failed = self._call(stage, payload,
                                   lambda r: validate_tool(stage, r, self.config.labels, plan["candidates"]), image)
        records[stage] = {"usable": False} if failed else result
        return failed

    @staticmethod
    def _neighbor_projection(case):
        return [{"ordinal": i + 1, "cosine": n.similarity, "labels": list(n.labels)}
                for i, n in enumerate(case.neighbors)]

    def prepare(self, case):
        if len(case.neighbors) != self.config.k:
            raise ValueError(f"Expected exactly {self.config.k} retrieved neighbors")
        if case.image is not None and not Path(case.image).is_file():
            raise ValueError("The private query image does not exist")
        if case.clinical_context is not None and not isinstance(case.clinical_context, (str, dict)):
            raise ValueError("Clinical context must be text or a structured record")
        state = build_state(case.neighbors, self.config.labels, self.knowledge)
        allowed = admissible_labels(state, self.config.labels)
        if set(allowed) - self.knowledge.keys():
            raise ValueError("Supply an entry for every admitted differential hypothesis")
        expanded = {c: self.knowledge[c] for c in allowed}
        return state, allowed, expanded

    def run(self, case: Case):
        self.events = []
        state, allowed, expanded = self.prepare(case)
        dv = disagreement(state.votes, state.raised)
        available = {"image": case.image is not None, "clinical_context": bool(case.clinical_context)}
        neighbors = self._neighbor_projection(case)
        # Only nondiagnostic availability flags enter RCMP; no path, raw image or text.
        planning = {"modality": case.modality, "labels": list(self.config.labels),
                    "available_views": available,
                    "retrieval_state": {"r": state.support, "q": state.votes,
                                        "disagreement": dv, "grounding": state.grounding},
                    "neighbors": neighbors}
        plan, plan_failed = self._call("RCMP", planning, lambda r: validate_plan(r, self.config.labels))
        if plan_failed:
            plan = {"case_complexity": "low", "view_priority": [], "candidates": list(state.raised),
                    "unresolved": [], "request_counterfactual": False,
                    "rationale": "Default equal-priority base plan; deep stages disabled"}
        dropped = [c for c in plan["candidates"] if c not in allowed]
        plan["candidates"] = [c for c in plan["candidates"] if c in allowed]
        plan["unresolved"] = [u for u in plan["unresolved"] if u["hypothesis"] in allowed]
        plan["view_priority"] = [v for v in plan["view_priority"] if available[v]]
        targets = {"candidates": plan["candidates"], "verification_targets": plan["unresolved"]}
        records, fallbacks = {}, ["RCMP"] if plan_failed else []

        def tool(stage, payload, image=None):
            if self._tool(stage, payload, plan, records, image):
                fallbacks.append(stage)

        if available["image"]:
            tool("T2", {**targets, "modality": case.modality}, image=case.image)
        if available["clinical_context"]:
            tool("T3", {**targets, "clinical_context": case.clinical_context})
        if all(records.get(t, {}).get("usable") for t in ("T2", "T3")):
            tool("T4", {"T2": records["T2"], "T3": records["T3"]})
        tool("T5", {"query_records": {t: records[t] for t in ("T2", "T3") if t in records},
                    "neighbors": neighbors})
        missing = [view for view in self.config.required_views
                   if not records.get({"image": "T2", "clinical_context": "T3"}[view], {}).get("usable")]
        cross = records.get("T4", {}).get("conflict_score", 0.0)
        gate = routing_gate(dv, cross, missing, plan["case_complexity"], self.config)
        # The manuscript's default-plan ablation has no deep stages, even if other gates fire.
        if gate["open"] and not plan_failed:
            tool("T6", {"candidates": plan["candidates"], "records": dict(records), "grounding": expanded})
            if plan["request_counterfactual"]:
                tool("T7", {"candidates": plan["candidates"], "records": dict(records), "grounding": expanded})
        usable = {t for t, record in records.items() if record.get("usable")}
        # Validated structured records, plan and G+ only. Never raw views or neighborhood.
        consensus = {"labels": list(self.config.labels), "plan": plan,
                     "records": dict(records), "grounding": expanded}
        ecc, ecc_failed = self._call("ECC", consensus,
                                     lambda r: validate_ecc(r, self.config.labels, usable))
        score_source = "ECC"
        if ecc_failed:
            fallbacks.append("ECC")
            score_source = "READOUT"
            ecc, readout_failed = self._call("READOUT", {"labels": list(self.config.labels),
                                                        "context": consensus},
                                             lambda r: validate_scores(r, self.config.labels))
            if readout_failed:
                fallbacks.append("READOUT")
                score_source = "neutral"
                ecc = {"ranked_candidates": [{"label": c, "score": 0.5} for c in self.config.labels]}
        scores = {item["label"]: item["score"] for item in ecc["ranked_candidates"]}
        fused = fuse(scores, state, self.config)
        return {"retrieval_state": asdict(state), "vote_disagreement": dv,
                "admissible_labels": list(allowed), "expanded_grounding_labels": list(expanded),
                "plan": plan, "dropped_candidates": dropped, "gate": gate,
                "missing_required_views": missing, "executed_tools": list(records),
                "records": records, "ecc": ecc, "score_source": score_source,
                "ecc_scores": scores, "fusion_weights": fused["alpha"],
                "final_scores": fused["scores"], "predictions": fused["predictions"],
                "fallbacks": fallbacks, "degraded": bool(fallbacks),
                "model_attempts": len(self.events), "model_calls": sum(e["attempt"] == 1 for e in self.events),
                "call_events": list(self.events)}
