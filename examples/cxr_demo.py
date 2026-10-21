"""Run one user-owned CXR locally. This script includes no case or image data."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from remedy import Case, MethodConfig, Neighbor, Pipeline
from remedy.model import HTTPBackend
from remedy.state import disagreement

REPO = Path(__file__).resolve().parents[1]


def private_path(value):
    path = Path(value).expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Keep private inputs and generated outputs outside the repository")
    return path


def read_json(path):
    return json.loads(private_path(path).read_text(encoding="utf-8-sig"))


def neighbors_from_json(items):
    if not isinstance(items, list):
        raise ValueError("Neighbors must be an array")
    return tuple(Neighbor(n.get("similarity", n.get("cos")), tuple(n["labels"])) for n in items)


def load_case(args, config):
    if args.case:
        manifest = private_path(args.case)
        raw = read_json(manifest)
        if raw.get("_cosines_are_reconstructed") or raw.get("_cosines_stubbed"):
            raise ValueError("Use measured retrieval similarities, not reconstructed/example cosines")
        views = raw.get("views", {})
        image = Path(views["image"])
        image = private_path(image if image.is_absolute() else manifest.parent / image)
        context = views.get("clinical_context")
        if args.neighbors or args.memory or args.context:
            raise ValueError("Use the case manifest or individual inputs, not both")
        neighbors = neighbors_from_json(raw["neighbors"])
        # All reference-label, subject, case-id and report fields are deliberately ignored.
    else:
        image = private_path(args.image)
        context = private_path(args.context).read_text(encoding="utf-8-sig") if args.context else None
        if args.neighbors:
            neighbors = neighbors_from_json(read_json(args.neighbors))
        elif args.memory:
            from remedy.retrieval import CosineMemory, encode_cxr
            if args.query_embedding:
                import numpy as np
                query = np.load(private_path(args.query_embedding), allow_pickle=False)
            else:
                query = encode_cxr(image, device=args.device)
            neighbors = CosineMemory.load(private_path(args.memory)).search(
                query, config.k, exclude_ids=args.exclude_memory_id)
        else:
            raise ValueError("Supply private --neighbors or a private --memory index")
    return Case(neighbors=neighbors, image=image, clinical_context=context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", help="Private case manifest compatible with the original runner")
    source.add_argument("--image", help="Private frontal query radiograph")
    parser.add_argument("--context", help="Private pre-test indication text file; never a findings/impression report")
    retrieval = parser.add_mutually_exclusive_group()
    retrieval.add_argument("--neighbors", help="Private top-K neighborhood JSON")
    retrieval.add_argument("--memory", help="Private BiomedCLIP NPZ index (emb, labels, optional ids)")
    parser.add_argument("--query-embedding", help="Optional private NPY query vector from the same frozen encoder")
    parser.add_argument("--exclude-memory-id", action="append", default=[])
    parser.add_argument("--knowledge", required=True, help="Private knowledge JSON keyed by direct label")
    parser.add_argument("--output", required=True, help="Private result JSON outside this repository")
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and CKM; make no model request")
    args = parser.parse_args(argv)
    try:
        if args.query_embedding and not args.memory:
            raise ValueError("--query-embedding requires --memory")
        if args.exclude_memory_id and not args.memory:
            raise ValueError("Memory exclusions require --memory")
        output = private_path(args.output)
        if output.suffix.lower() != ".json":
            raise ValueError("Use a private .json output file")
        knowledge = read_json(args.knowledge)
        config = MethodConfig(k=args.k)
        case = load_case(args, config)
        backend = None if args.dry_run else HTTPBackend.from_env()
        pipeline = Pipeline(backend, knowledge, config)
        if args.dry_run:
            state, allowed, expanded = pipeline.prepare(case)
            result = {"mode": "input_validation_only", "model_attempts": 0,
                      "retrieval_state": asdict(state),
                      "vote_disagreement": disagreement(state.votes, state.raised),
                      "admissible_labels": list(allowed), "expanded_grounding_labels": list(expanded)}
        else:
            result = {"mode": "model_inference", **pipeline.run(case)}
            result["usage"] = backend.usage
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids overwriting an existing private result or input file.
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"mode": result["mode"], "model_attempts": result["model_attempts"],
                          "degraded": result.get("degraded", False),
                          "output_written": True}))
        return 2 if result.get("degraded") else 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        # Do not echo a private manifest, path or backend response onto a shared terminal.
        print(f"CXR run could not complete ({type(error).__name__}). Check private input paths, schema and endpoint settings.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
