# REMEDY

Core implementation of **REMEDY: Persistent retrieval-state control for training-free multimodal medical inference**.

The release contains the inference core and a command-line CXR demonstration. **No datasets, chest radiographs, patient records, case manifests, retrieved exemplars, embeddings, knowledge-base contents, model weights or inference outputs are distributed.** Supply all private resources locally, outside the checkout.

## What is implemented

- **CKM**: nonnegative cosine affinities, similarity-weighted support `r(x)`, unweighted neighbor votes `q(x)` and retrieval-conditioned grounding.
- **RCMP**: planning without raw diagnostic views; deterministic candidate admission through retrieved labels and named differential links.
- **DDTR**: independently recorded disagreement, evidence and planner gates; gated T6 and planner-requested T7.
- **ECC**: five adjudication rules, validated structured evidence and a complete closed-label score vector.
- **RSF**: per-label leave-one-label-out agreement weighting, followed by thresholding.
- Explicit retries and fallbacks, view separation, local retrieval, and a private-input CXR CLI.

This release contains the core inference code and a private-input CXR demonstration. It does not reproduce the paper's benchmark tables by itself. No empirical performance claim is established by the included tests.

## Install and test

Python 3.10 or newer:

```bash
python -m pip install -e '.[retrieval]'
python -m unittest discover -s tests -v
python tools/check_release.py
```

The core uses the Python standard library. NumPy is optional for local embedding retrieval. To encode a new image with the frozen BiomedCLIP image tower:

```bash
python -m pip install -e '.[vision]'
```

That optional path loads the pretrained `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` model through `open_clip`; prepare an authorized local cache for offline use. No model weight is included here.

## Run a CXR demonstration with your own private inputs

Use a frontal radiograph, pre-test indication text, a private CheXpert-memory neighborhood and private clinical grounding. Do not pass the target report's FINDINGS/IMPRESSION or reference diagnosis to the model. Keep source memory and target evaluation cases disjoint.

First check the inputs without making a model request:

```bash
python examples/cxr_demo.py \
  --image /private/remedy/query.jpg \
  --context /private/remedy/indication.txt \
  --neighbors /private/remedy/neighbors.json \
  --knowledge /private/remedy/knowledge.json \
  --output /private/remedy/check-result.json \
  --dry-run
```

All paths above are user-supplied private paths; none of those files are bundled. The CLI refuses inputs and outputs inside this checkout. It creates a new result file and never overwrites an existing one.

Configure a VLM endpoint authorized for your private data. For a locally hosted OpenAI-compatible model:

```bash
export REMEDY_BASE_URL=http://127.0.0.1:8000/v1
export REMEDY_MODEL=Qwen/Qwen3-VL-8B-Instruct
export REMEDY_API_KEY=
```

PowerShell equivalents use `$env:REMEDY_BASE_URL`, `$env:REMEDY_MODEL` and `$env:REMEDY_API_KEY`. The `.env.example` file is a reference; the program does not automatically load an environment file.

For actual model inference, run the same command without `--dry-run`, choosing a fresh output filename. The image is sent only to the explicitly configured endpoint and only in T2. Loopback endpoints bypass system proxies; remote endpoints retain normal proxy settings. The backend does not save prompts, images or raw completions to logs. Returned evidence and scores remain private outputs and must not be committed.

Exit codes: `0` completed, `1` invalid inputs/setup, `2` inference completed with at least one explicit fallback. A fallback run must not be reported as a successful model evaluation.

### Private input formats

`neighbors.json` is an array of exactly K objects. Each object has `similarity` (or the legacy `cos`) containing a real cosine in `[-1,1]`, and `labels` containing a list of mapped direct labels. Empty label lists are allowed. Do not substitute missing cosines. The default K is 5. Manifests marked as reconstructed or stubbed retrieval are rejected.

`knowledge.json` maps each required label to an object with `criteria`, optional `concepts` and `references`, and a `differentials` list of named direct labels. Entries are required for retrieval-raised labels and their admitted differential hypotheses. The legacy `concept` and `standard` fields are accepted. The knowledge-base contents are private inputs, not part of this repository.

Alternatively, use a private manifest from the original runner:

```bash
python examples/cxr_demo.py \
  --case /private/remedy/case.json \
  --knowledge /private/remedy/knowledge.json \
  --output /private/remedy/run-result.json
```

Only `neighbors`, `views.image` and `views.clinical_context` are read. Reference labels and patient identifiers are ignored. Relative image paths resolve against the private manifest directory.

### Retrieve neighbors locally

Replace `--neighbors` with `--memory /private/remedy/memory.npz`. The index contains normalized or unnormalized floating-point `emb` rows and fixed-width Unicode `labels` rows, with labels separated by `|`. An optional Unicode `ids` array enables `--exclude-memory-id`. NumPy object arrays are deliberately rejected.

Without `--query-embedding`, this path encodes the image with frozen BiomedCLIP and requires the `vision` extra. With `--query-embedding /private/remedy/query.npy`, it uses a precomputed vector from the **same encoder and preprocessing** and requires only the `retrieval` extra. The repository contains no example index or embeddings.

## Layout

```text
src/remedy/          core arithmetic, stage graph, prompts, validation and model adapter
examples/cxr_demo.py one private-input CXR entry point
tests/              mathematical and fake-backend tests; no clinical fixtures
docs/METHOD.md      method correspondence and implementation boundaries
tools/check_release.py  tracked-file publication check
```

See [method correspondence](docs/METHOD.md) for equations and fallback behavior. The MIT license applies to the code; it grants no rights to private datasets or third-party model/terminology resources. This software is for research, not clinical deployment.
