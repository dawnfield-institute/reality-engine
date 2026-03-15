"""ParticleAnalyzer — classifies detected structures by mass/charge/spin analogs."""

from __future__ import annotations

from typing import List

from src.v3.analyzers.base import Detection


class ParticleAnalyzer:
    """Classify detections into particle-like categories based on properties."""

    def classify(self, detections: List[Detection]) -> List[Detection]:
        """Add particle classification to existing detections.

        Classifies based on mass and energy properties:
        - light (< 0.5 mass) → "lepton-like"
        - medium (0.5-2.0) → "meson-like"
        - heavy (> 2.0) → "baryon-like"
        """
        for d in detections:
            if d.kind not in ("atom", "gravity_well", "star"):
                continue
            mass = d.properties.get("mass", 0)
            if mass < 0.5:
                d.properties["particle_class"] = "lepton-like"
            elif mass < 2.0:
                d.properties["particle_class"] = "meson-like"
            else:
                d.properties["particle_class"] = "baryon-like"
        return detections
