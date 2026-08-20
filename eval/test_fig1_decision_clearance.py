# -*- coding: utf-8 -*-
"""Regression check for text crossing the decision-output arrows."""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
MODULE_UNDER_TEST = os.environ.get(
    "FIG1_MODULE_UNDER_TEST", "make_fig1_v14_edge_clean"
)


def segment_crosses_box(
    start: tuple[float, float],
    end: tuple[float, float],
    box: tuple[float, float, float, float],
    margin: float = 0.20,
) -> bool:
    x = np.linspace(start[0], end[0], 801)
    y = np.linspace(start[1], end[1], 801)
    x0, y0, x1, y1 = box
    return bool(
        np.any(
            (x >= x0 - margin)
            & (x <= x1 + margin)
            & (y >= y0 - margin)
            & (y <= y1 + margin)
        )
    )


class DecisionClearanceTest(unittest.TestCase):
    def test_output_arrows_do_not_cross_text(self) -> None:
        module = importlib.import_module(MODULE_UNDER_TEST)
        module.assemble()
        base = module.hierarchical.modular.base
        base.fig.canvas.draw()
        renderer = base.fig.canvas.get_renderer()
        inverse = base.ax.transData.inverted()
        arrow_segments = (
            ((157.8, 41.2), (148.0, 25.2)),
            ((162.2, 41.2), (171.0, 25.2)),
        )
        collisions: list[str] = []

        for artist in base.ax.texts:
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            bounds = artist.get_window_extent(renderer=renderer).transformed(inverse)
            box = (bounds.x0, bounds.y0, bounds.x1, bounds.y1)
            if any(segment_crosses_box(start, end, box) for start, end in arrow_segments):
                collisions.append(artist.get_text().replace("\n", " / "))

        plt.close(base.fig)
        self.assertFalse(collisions, "Decision arrows cross text: " + "; ".join(collisions))


if __name__ == "__main__":
    unittest.main()
