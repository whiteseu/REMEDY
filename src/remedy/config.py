"""Method settings selected on source development data in the manuscript."""

from dataclasses import dataclass
import math

CXR_LABELS = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
)


@dataclass(frozen=True)
class MethodConfig:
    labels: tuple[str, ...] = CXR_LABELS
    k: int = 5
    alpha0: float = 0.3
    tau_vote: float = 0.4
    tau_cross: float = 0.5
    delta: float = 0.5
    required_views: tuple[str, ...] = ("image", "clinical_context")

    def __post_init__(self):
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("Direct labels must be nonempty and unique")
        if any(not isinstance(c, str) or not c.strip() for c in self.labels):
            raise ValueError("Labels must be nonempty strings")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("k must be a positive integer")
        for name in ("alpha0", "tau_vote", "tau_cross", "delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if set(self.required_views) - {"image", "clinical_context"}:
            raise ValueError("This release supports image and clinical_context views")
