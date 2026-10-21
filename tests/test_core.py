"""Mathematical fixtures and fake model records only; no medical case data."""

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

from remedy import Case, MethodConfig, Neighbor, Pipeline
from remedy.model import ModelError, parse_object
from remedy.schemas import validate_scores
from remedy.state import (admissible_labels, build_state, derived_no_mapped_finding,
                          disagreement, fuse, leave_one_out_disagreement, routing_gate)


LABELS = ("A", "B", "C")


def knowledge():
    return {c: {"criteria": "toy criterion " + c, "differentials": ["B"] if c == "A" else []} for c in LABELS}


class FakeBackend:
    def __init__(self, high=False, counterfactual=False, failures=(), responses=None):
        self.high, self.counterfactual = high, counterfactual
        self.failures = set(failures)
        self.responses = responses or {}
        self.seen = []

    def complete(self, stage, system, payload, image=None):
        self.seen.append((stage, deepcopy(payload), image, system))
        if stage in self.failures:
            raise ModelError("test failure")
        if stage in self.responses:
            value = self.responses[stage]
            return value() if callable(value) else deepcopy(value)
        if stage == "RCMP":
            return {"case_complexity": "high" if self.high else "low", "view_priority": ["image", "clinical_context"],
                    "candidates": ["A", "B", "C", "not-in-vocabulary"],
                    "unresolved": [{"hypothesis": c, "evidence_needed": "toy evidence"} for c in LABELS],
                    "request_counterfactual": self.counterfactual, "rationale": "toy plan"}
        if stage in {"T2", "T3"}:
            return {"usable": True, "findings": [], "supporting_labels": [], "opposing_labels": []}
        if stage == "T4":
            return {"usable": True, "conflict_score": 0, "conflicts": [], "more_reliable_view": "tie"}
        if stage == "T5":
            return {"usable": True, "label_support": {"A": 1}, "label_distribution": {"A": 1}}
        if stage == "T6":
            return {"usable": True, "ranking": [], "pairwise": []}
        if stage == "T7":
            return {"usable": True, "per_candidate": []}
        if stage in {"ECC", "READOUT"}:
            return {"ranked_candidates": [{"label": c, "score": 0.8 if c == "C" else 0.2} for c in LABELS],
                    "evidence_chain": [{"label": c, "tool": "T2", "anchor": "toy normal read"} for c in LABELS],
                    "counter_evidence": [], "confidence": 0.7}
        raise AssertionError(stage)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MethodConfig(labels=LABELS)

    def test_similarity_weighting_and_vote_disagreement_are_separate(self):
        n = [Neighbor(0.9, ("A",)), Neighbor(0.1, ("B",))]
        s = build_state(n, LABELS, knowledge())
        self.assertAlmostEqual(s.support["A"], 0.9)
        self.assertEqual(s.votes["A"], 0.5)
        self.assertEqual(disagreement(s.votes, s.raised), 1)
        self.assertNotEqual(disagreement(s.support, s.raised), 1)

    def test_nonpositive_affinities_use_uniform_weights(self):
        s = build_state([Neighbor(-0.2, ("A",)), Neighbor(0, ("B",))], LABELS, knowledge())
        self.assertEqual(s.weights, (0.5, 0.5))

    def test_no_raised_label_and_empty_loo_are_zero(self):
        s = build_state([Neighbor(1, ())], LABELS, knowledge())
        self.assertEqual(s.raised, ())
        self.assertEqual(disagreement(s.votes, s.raised), 0)
        self.assertEqual(leave_one_out_disagreement(s, "A"), 0)
        one = build_state([Neighbor(1, ("A",))], LABELS, knowledge())
        self.assertEqual(leave_one_out_disagreement(one, "A"), 0)

    def test_loo_uses_other_raised_coordinates_and_does_not_include_unraised(self):
        s = build_state([Neighbor(0.8, ("A", "B")), Neighbor(0.2, ("A",))], LABELS, knowledge())
        self.assertEqual(leave_one_out_disagreement(s, "A"), 1)
        self.assertEqual(leave_one_out_disagreement(s, "B"), 0)
        result = fuse({"A": 0.7, "B": 0.8, "C": 0.9}, s, self.cfg)
        self.assertAlmostEqual(result["alpha"]["A"], 0)
        self.assertAlmostEqual(result["alpha"]["B"], 0.3)
        self.assertAlmostEqual(result["alpha"]["C"], 0.15)
        self.assertAlmostEqual(result["scores"]["A"], 0.7)

    def test_paper_loo_arithmetic_without_patient_fixture(self):
        s = build_state([Neighbor(1, ("A", "B") if i < 2 else ("A",)) for i in range(5)], LABELS, knowledge())
        # On label C: mean disagreement on A (0) and B (.96) is .48.
        result = fuse({"A": 0.5, "B": 0.5, "C": 0.82}, s, self.cfg)
        self.assertAlmostEqual(result["alpha"]["C"], 0.156)
        self.assertAlmostEqual(result["scores"]["C"], 0.69208)

    def test_candidates_only_expand_through_named_links(self):
        s = build_state([Neighbor(1, ("A",))], LABELS, knowledge())
        self.assertEqual(admissible_labels(s, LABELS), ("A", "B"))

    def test_bad_cosines_empty_memory_and_unknown_labels_fail(self):
        for val in (float("nan"), float("inf"), 1.1, True):
            with self.assertRaises(ValueError):
                Neighbor(val, ())
        with self.assertRaises(ValueError):
            build_state([], LABELS, knowledge())
        with self.assertRaises(ValueError):
            build_state([Neighbor(1, ("unknown",))], LABELS, knowledge())

    def test_scores_are_complete_finite_and_bounded(self):
        s = build_state([Neighbor(1, ("A",))], LABELS, knowledge())
        for scores in ({"A": 0.1}, {"A": 0.1, "B": math.nan, "C": 0.2},
                       {"A": 0.1, "B": 0.2, "C": 2}):
            with self.assertRaises(ValueError):
                fuse(scores, s, self.cfg)

    def test_gate_sources_and_inclusive_thresholds(self):
        for args, source in [((0.4, 0, [], "low"), "state"), ((0, 0.5, [], "low"), "evidence"),
                             ((0, 0, ["image"], "low"), "evidence"), ((0, 0, [], "high"), "plan")]:
            gate = routing_gate(*args, self.cfg)
            self.assertTrue(gate["open"] and gate[source])

    def test_derived_coordinate_strict_threshold_tie(self):
        self.assertEqual(derived_no_mapped_finding({"A": 0.5}), (0.5, False))
        self.assertEqual(derived_no_mapped_finding({"A": 0.3}), (0.7, True))

    def test_schema_rejects_duplicate_and_missing_scores(self):
        for items in ([{"label": "A", "score": 0.2}],
                      [{"label": "A", "score": 0.2}] * 3):
            with self.assertRaises(ValueError):
                validate_scores({"ranked_candidates": items}, LABELS)
        with self.assertRaises(ValueError):
            parse_object('{"score":0,"score":1}')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "view.png"
        self.image.write_bytes(b"test-view-for-fake-backend")
        self.case = Case(tuple(Neighbor(1, ("A",)) for _ in range(5)), self.image, "PRIVATE_VIEW_SENTINEL")
        self.cfg = MethodConfig(labels=LABELS)

    def run_pipeline(self, backend, case=None):
        return Pipeline(backend, knowledge(), self.cfg).run(case or self.case)

    def test_pruned_path_six_calls_and_view_isolation(self):
        b = FakeBackend()
        r = self.run_pipeline(b)
        self.assertEqual(r["model_attempts"], 6)
        self.assertEqual(r["executed_tools"], ["T2", "T3", "T4", "T5"])
        self.assertFalse(r["degraded"])
        for stage, payload, image, _ in b.seen:
            self.assertEqual(image is not None, stage == "T2")
            self.assertEqual("PRIVATE_VIEW_SENTINEL" in str(payload), stage == "T3")
            self.assertEqual("neighbors" in payload, stage in {"RCMP", "T5"})

    def test_full_path_eight_calls_and_expanded_grounding(self):
        b = FakeBackend(high=True, counterfactual=True)
        r = self.run_pipeline(b)
        self.assertEqual(r["model_attempts"], 8)
        self.assertEqual(r["plan"]["candidates"], ["A", "B"])
        self.assertEqual(r["dropped_candidates"], ["C", "not-in-vocabulary"])
        for stage, payload, _, _ in b.seen:
            if stage in {"T6", "T7", "ECC"}:
                self.assertEqual(set(payload["grounding"]), {"A", "B"})
        self.assertIn("C", r["ecc_scores"])  # Candidate admission never narrows the scored space.

    def test_gate_open_without_requested_t7_only_admits_t6(self):
        r = self.run_pipeline(FakeBackend(high=True))
        self.assertIn("T6", r["executed_tools"])
        self.assertNotIn("T7", r["executed_tools"])
        self.assertEqual(r["model_attempts"], 7)

    def test_counterfactual_request_alone_does_not_open_gate(self):
        r = self.run_pipeline(FakeBackend(counterfactual=True))
        self.assertFalse(r["gate"]["open"])
        self.assertNotIn("T7", r["executed_tools"])

    def test_failed_view_is_unusable_and_opens_missing_evidence_gate(self):
        r = self.run_pipeline(FakeBackend(failures={"T2"}))
        self.assertEqual(r["records"]["T2"], {"usable": False})
        self.assertIn("image", r["missing_required_views"])
        self.assertTrue(r["gate"]["evidence"])
        self.assertIn("T6", r["executed_tools"])
        self.assertNotIn("T4", r["executed_tools"])

    def test_readable_normal_image_is_not_missing(self):
        r = self.run_pipeline(FakeBackend())
        self.assertEqual(r["records"]["T2"]["findings"], [])
        self.assertEqual(r["missing_required_views"], [])

    def test_missing_context_is_skipped_but_gate_escalates(self):
        r = self.run_pipeline(FakeBackend(), replace(self.case, clinical_context=None))
        self.assertNotIn("T3", r["executed_tools"])
        self.assertIn("clinical_context", r["missing_required_views"])
        self.assertIn("T6", r["executed_tools"])

    def test_default_planner_fallback_disables_deep_stages(self):
        r = self.run_pipeline(FakeBackend(failures={"RCMP", "T2"}))
        self.assertIn("RCMP", r["fallbacks"])
        self.assertTrue(r["gate"]["open"])
        self.assertNotIn("T6", r["executed_tools"])

    def test_ecc_failure_uses_shared_readout_not_retrieval_scores(self):
        r = self.run_pipeline(FakeBackend(failures={"ECC"}))
        self.assertEqual(r["score_source"], "READOUT")
        self.assertEqual(r["ecc_scores"]["C"], 0.8)
        self.assertIn("ECC", r["fallbacks"])

    def test_all_readouts_fail_neutral_scores_are_explicit(self):
        r = self.run_pipeline(FakeBackend(failures={"ECC", "READOUT"}))
        self.assertEqual(r["score_source"], "neutral")
        self.assertEqual(set(r["ecc_scores"].values()), {0.5})
        self.assertTrue(r["degraded"])

    def test_malformed_nested_strength_triggers_retry_and_empty_record(self):
        bad = {"usable": True, "findings": [], "opposing_labels": [],
               "supporting_labels": [{"label": "A", "strength": float("nan"), "anchor": "toy"}]}
        r = self.run_pipeline(FakeBackend(responses={"T2": bad}))
        self.assertIn("T2", r["fallbacks"])
        self.assertEqual(len([e for e in r["call_events"] if e["stage"] == "T2"]), 2)

    def test_unexecuted_ecc_anchor_is_rejected(self):
        b = FakeBackend()
        raw = b.complete("ECC", "", {})
        raw["evidence_chain"] = [{"label": "A", "tool": "T7", "anchor": "invented"}]
        r = self.run_pipeline(FakeBackend(responses={"ECC": raw}))
        self.assertEqual(r["score_source"], "READOUT")

    def test_ecc_requires_an_anchor_for_each_scored_label(self):
        raw = FakeBackend().complete("ECC", "", {})
        raw["evidence_chain"] = raw["evidence_chain"][:1]
        r = self.run_pipeline(FakeBackend(responses={"ECC": raw}))
        self.assertEqual(r["score_source"], "READOUT")

    def test_expanded_knowledge_must_exist_before_any_model_call(self):
        b = FakeBackend()
        with self.assertRaises(ValueError):
            Pipeline(b, {"A": knowledge()["A"]}, self.cfg).run(self.case)
        self.assertEqual(b.seen, [])

    def test_reuse_does_not_accumulate_other_case_events(self):
        pipeline = Pipeline(FakeBackend(), knowledge(), self.cfg)
        first = pipeline.run(self.case)
        second = pipeline.run(self.case)
        self.assertEqual(first["model_attempts"], second["model_attempts"])
        self.assertEqual(len(first["call_events"]), 6)


if __name__ == "__main__":
    unittest.main()
