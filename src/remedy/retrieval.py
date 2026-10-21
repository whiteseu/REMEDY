"""Optional local cosine retrieval over private, precomputed memory embeddings."""

import math
from pathlib import Path
from .state import Neighbor


class CosineMemory:
    def __init__(self, embeddings, label_sets, ids=None):
        import numpy as np
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("Memory must be a nonempty finite embedding matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms == 0).any() or len(label_sets) != len(matrix):
            raise ValueError("Memory has a zero vector or mismatched labels")
        self.embeddings = matrix / norms
        self.label_sets = [tuple(labels) for labels in label_sets]
        self.ids = list(ids) if ids is not None else None
        if self.ids is not None and (len(self.ids) != len(matrix) or len(set(self.ids)) != len(self.ids)):
            raise ValueError("Memory ids must be unique and match the embedding rows")

    @classmethod
    def load(cls, path):
        import numpy as np
        # Deliberately reject object arrays / pickle-bearing legacy archives.
        with np.load(Path(path), allow_pickle=False) as data:
            embeddings = data["emb"]
            labels = [str(value).split("|") if str(value) else [] for value in data["labels"]]
            ids = [str(value) for value in data["ids"]] if "ids" in data.files else None
            return cls(embeddings, labels, ids)

    def search(self, query, k=5, exclude_ids=()):
        import numpy as np
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (self.embeddings.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("Query embedding shape/values do not match the memory")
        norm = np.linalg.norm(vector)
        if not math.isfinite(float(norm)) or norm == 0:
            raise ValueError("Query embedding must have nonzero finite norm")
        scores = self.embeddings @ (vector / norm)
        eligible = np.ones(len(scores), dtype=bool)
        if exclude_ids:
            if self.ids is None:
                raise ValueError("Exclusion requires a memory index with ids")
            blocked = set(exclude_ids)
            eligible = np.array([item not in blocked for item in self.ids])
        if isinstance(k, bool) or not isinstance(k, int) or k < 1 or int(eligible.sum()) < k:
            raise ValueError("k exceeds the eligible memory size")
        indices = np.flatnonzero(eligible)
        # Stable sorting makes equal-similarity retrieval deterministic.
        top = indices[np.argsort(-scores[indices], kind="stable")[:k]]
        return tuple(Neighbor(float(np.clip(scores[i], -1, 1)), self.label_sets[i]) for i in top)


def encode_cxr(image: Path, device=None):
    """Frozen BiomedCLIP image tower, loaded only when this optional path is requested."""
    import open_clip
    import torch
    from PIL import Image
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with Image.open(image) as im:
        tensor = preprocess(im.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        vector = model.encode_image(tensor).float().cpu().numpy()[0]
    return vector
