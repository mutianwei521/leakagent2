# -*- coding: utf-8 -*-
"""Regression checks for the hard-contract checklist bounds."""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
MODULE_UNDER_TEST = os.environ.get(
    "FIG1_MODULE_UNDER_TEST", "make_fig1_v15_clearance"
)


class ContractBoundsTest(unittest.TestCase):
    def test_checklist_and_identifier_stay_inside_contract_card(self) -> None:
        module = importlib.import_module(MODULE_UNDER_TEST)
        module.assemble()
        base = module.hierarchical.modular.base
        base.fig.canvas.draw()
        renderer = base.fig.canvas.get_renderer()
        inverse = base.ax.transData.inverted()

        identifiers = [artist for artist in base.ax.texts if artist.get_text() == "G1–G7"]
        self.assertEqual(len(identifiers), 1)
        text_box = identifiers[0].get_window_extent(renderer=renderer).transformed(inverse)
        self.assertGreaterEqual(text_box.x0, 142.5)
        self.assertLessEqual(text_box.x1, 157.5)
        self.assertGreaterEqual(text_box.y0, 62.5)
        self.assertLessEqual(text_box.y1, 68.0)

        checkboxes = []
        for patch in base.ax.patches:
            if not isinstance(patch, Rectangle):
                continue
            x, y = patch.get_xy()
            if 144.0 <= x <= 147.0 and 60.0 <= y <= 80.0 and patch.get_width() <= 2.0:
                checkboxes.append(patch)

        self.assertGreaterEqual(len(checkboxes), 3)
        for checkbox in checkboxes:
            _, y = checkbox.get_xy()
            self.assertGreaterEqual(
                y,
                68.0,
                "Checklist row intrudes into the G1–G7 label or exits the card",
            )

        plt.close(base.fig)


if __name__ == "__main__":
    unittest.main()
