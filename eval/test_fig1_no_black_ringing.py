# -*- coding: utf-8 -*-
"""Regression check for neutral black ringing around Fig. 1 raster assets."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURE = ROOT / "figures" / "Fig1_architecture_v13_hierarchical.png"
FIGURE = Path(os.environ.get("FIG1_UNDER_TEST", DEFAULT_FIGURE))

# Data-coordinate regions containing only the five generated hero assets.
ROIS = {
    "network": (5.3, 64.0, 31.7, 88.5),
    "twin": (64.5, 69.0, 84.5, 83.6),
    "dossier": (115.5, 59.5, 132.5, 88.1),
    "chip": (162.6, 68.0, 175.4, 82.0),
    "active sensing": (76.8, 9.0, 86.4, 21.8),
}


class BlackRingingTest(unittest.TestCase):
    def test_assets_have_no_neutral_black_ringing(self) -> None:
        self.assertTrue(FIGURE.exists(), f"Missing figure: {FIGURE}")
        rgb = np.asarray(Image.open(FIGURE).convert("RGB"), dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]
        failures: list[str] = []

        for name, (x0, y0, x1, y1) in ROIS.items():
            roi = rgb[
                round((112.0 - y1) / 112.0 * height) : round((112.0 - y0) / 112.0 * height),
                round(x0 / 183.0 * width) : round(x1 / 183.0 * width),
            ]
            luma = 0.2126 * roi[..., 0] + 0.7152 * roi[..., 1] + 0.0722 * roi[..., 2]
            chroma = roi.max(axis=2) - roi.min(axis=2)
            neutral_black = (luma < 0.12) & (chroma < 0.18)
            count = int(np.count_nonzero(neutral_black))
            allowed = max(20, round(neutral_black.size * 0.0002))
            if count > allowed:
                failures.append(f"{name}: {count} black pixels (allowed {allowed})")

        self.assertFalse(failures, "Interpolation ringing detected: " + "; ".join(failures))


if __name__ == "__main__":
    unittest.main()
