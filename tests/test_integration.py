"""Exercise HTTP serialization, private-input CLI and local retrieval without real cases."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from remedy import CXR_LABELS
from remedy.model import HTTPBackend, ModelError
from remedy.prompts import ROLES

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def provider(response_fn):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            status, result = response_fn(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def envelope(record):
    return {"choices": [{"message": {"content": json.dumps(record)}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def fake_stage_response(payload):
    prompt = payload["messages"][0]["content"]
    stage = next(s for s, role in ROLES.items() if role in prompt)
    if stage == "RCMP":
        result = {"case_complexity": "low", "view_priority": ["image", "clinical_context"],
                  "candidates": [CXR_LABELS[0]], "unresolved": [], "request_counterfactual": False,
                  "rationale": "protocol test"}
    elif stage in {"T2", "T3"}:
        result = {"usable": True, "findings": [], "supporting_labels": [], "opposing_labels": []}
    elif stage == "T4":
        result = {"usable": True, "conflict_score": 0, "conflicts": [], "more_reliable_view": "tie"}
    elif stage == "T5":
        result = {"usable": True, "label_support": {}, "label_distribution": {}}
    elif stage == "ECC":
        result = {"ranked_candidates": [{"label": c, "score": 0.2} for c in CXR_LABELS],
                  "evidence_chain": [{"label": c, "tool": "T2", "anchor": "protocol test"} for c in CXR_LABELS],
                  "counter_evidence": [], "confidence": 0.5}
    else:
        raise AssertionError(stage)
    return 200, envelope(result)


class TransportTests(unittest.TestCase):
    def test_loopback_endpoint_bypasses_system_proxy(self):
        with provider(lambda _: (200, envelope({"ok": True}))) as (url, seen):
            with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "https_proxy": "http://127.0.0.1:1",
                                         "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1",
                                         "no_proxy": "", "NO_PROXY": ""}):
                HTTPBackend(url, "test-model").complete("RCMP", "", {})
            self.assertEqual(len(seen), 1)

    def test_http_adapter_encodes_only_supplied_image_and_uses_zero_temperature(self):
        with tempfile.TemporaryDirectory() as tmp, provider(lambda _: (200, envelope({"ok": True}))) as (url, seen):
            image = Path(tmp) / "view.png"
            image.write_bytes(b"protocol-test-bytes")
            backend = HTTPBackend(url, "test-model")
            backend.complete("T2", "system", {"text": "unicode: 测试"}, image)
            self.assertEqual(seen[0]["temperature"], 0)
            self.assertEqual(len(seen[0]["messages"][1]["content"]), 2)
            self.assertEqual(backend.usage, [{"stage": "T2", "prompt_tokens": 3, "completion_tokens": 2}])
            self.assertNotIn("messages", str(backend.usage))

    def test_provider_error_body_is_not_exposed(self):
        with provider(lambda _: (401, {"private": "DO_NOT_ECHO_THIS_BODY"})) as (url, _):
            with self.assertRaises(ModelError) as caught:
                HTTPBackend(url, "test-model").complete("T2", "", {})
            self.assertNotIn("DO_NOT_ECHO", str(caught.exception))

    def test_retryable_transport_failure_recovers(self):
        count = [0]
        def reply(_):
            count[0] += 1
            return (503, {}) if count[0] == 1 else (200, envelope({"ok": True}))
        with provider(reply) as (url, _):
            HTTPBackend(url, "test-model").complete("RCMP", "", {})
        self.assertEqual(count[0], 2)


class DemoTests(unittest.TestCase):
    def test_private_cli_runs_full_stage_protocol_and_protects_outputs(self):
        with tempfile.TemporaryDirectory() as tmp, provider(fake_stage_response) as (url, seen):
            private = Path(tmp)
            image = private / "query.png"
            image.write_bytes(b"protocol-test-only")
            context = private / "context.txt"
            context.write_text("PRIVATE_CONTEXT_SENTINEL")
            neighbors = private / "neighbors.json"
            neighbors.write_text(json.dumps([{"cos": 1, "labels": [CXR_LABELS[0]]}] * 5))
            kb = private / "knowledge.json"
            kb.write_text(json.dumps({CXR_LABELS[0]: {"criteria": "test criterion", "differentials": []}}))
            output = private / "result.json"
            cmd = [sys.executable, str(ROOT / "examples/cxr_demo.py"), "--image", str(image),
                   "--context", str(context), "--neighbors", str(neighbors), "--knowledge", str(kb),
                   "--output", str(output)]
            env = {**os.environ, "REMEDY_BASE_URL": url, "REMEDY_MODEL": "test-model", "REMEDY_API_KEY": ""}
            run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(output.read_text())
            self.assertFalse(result["degraded"])
            self.assertEqual(result["model_attempts"], 6)
            self.assertEqual(len(result["final_scores"]), 14)
            self.assertNotIn(str(image), output.read_text())
            self.assertNotIn("PRIVATE_CONTEXT_SENTINEL", run.stdout)
            view_messages = [r for r in seen if len(r["messages"][1]["content"]) == 2]
            self.assertEqual(len(view_messages), 1)
            old = output.read_bytes()
            blocked = subprocess.run(cmd + ["--dry-run"], env=env, capture_output=True, timeout=30)
            self.assertEqual(blocked.returncode, 1)
            self.assertEqual(output.read_bytes(), old)

    def test_private_cli_refuses_data_under_checkout(self):
        cmd = [sys.executable, str(ROOT / "examples/cxr_demo.py"), "--image", str(ROOT / "query.jpg"),
               "--knowledge", str(ROOT / "knowledge.json"), "--output", str(ROOT / "result.json"), "--dry-run"]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertFalse((ROOT / "result.json").exists())


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Optional NumPy retrieval extra not installed")
class RetrievalTests(unittest.TestCase):
    def test_cosine_search_normalizes_and_excludes_ids(self):
        from remedy.retrieval import CosineMemory
        memory = CosineMemory([[2, 0], [1, 1], [0, 3]], [("A",), ("B",), ()], ["first", "second", "third"])
        result = memory.search([1, 0], k=2, exclude_ids=["first"])
        self.assertEqual(result[0].labels, ("B",))
        self.assertEqual(result[1].labels, ())

    def test_zero_or_nonfinite_query_and_oversize_k_fail(self):
        from remedy.retrieval import CosineMemory
        memory = CosineMemory([[1, 0]], [("A",)])
        for query, k in [([0, 0], 1), ([float("nan"), 0], 1), ([1, 0], 2)]:
            with self.assertRaises(ValueError):
                memory.search(query, k)

    def test_npz_archive_has_no_pickle_and_loads_local_only(self):
        import numpy as np
        from remedy.retrieval import CosineMemory
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "memory.npz"
            np.savez(file, emb=np.eye(2), labels=np.array(["A|B", ""]))
            self.assertEqual(CosineMemory.load(file).search([1, 0], 1)[0].labels, ("A", "B"))
            np.savez(file, emb=np.eye(2), labels=np.array([{}, {}], dtype=object))
            with self.assertRaises(ValueError):
                CosineMemory.load(file)


if __name__ == "__main__":
    unittest.main()
