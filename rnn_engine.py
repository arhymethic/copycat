"""
rnn_engine.py - Integration Adapter for sjvasquez/handwriting-synthesis (Alex Graves RNN).

Repository Reference:
https://github.com/sjvasquez/handwriting-synthesis

This module provides a clean bridge to the recurrent neural network (RNN)
handwriting synthesis engine when its weights and dependencies (TensorFlow)
are available.

If weights or dependencies are absent, it raises a clear informative exception
or falls back gracefully to the built-in single-stroke Hershey vector engine.
"""

import os
import sys
import logging
from typing import List, Tuple, Optional

# Type alias
Stroke = List[Tuple[float, float]]

logger = logging.getLogger("copycat.rnn")


class RNNStrokeEngine:
    """
    Wrapper around sjvasquez/handwriting-synthesis model.
    Synthesizes handwriting using Graves' Mixture Density RNN.
    """

    def __init__(self, checkpoints_dir: Optional[str] = None):
        self.checkpoints_dir = checkpoints_dir or os.path.join(
            os.path.dirname(__file__), "checkpoints"
        )
        self.hand = None
        self._initialize()

    def is_available(self) -> bool:
        """Check if RNN synthesizer can be loaded."""
        return self.hand is not None

    def _initialize(self):
        """Attempts to load the HandwritingSynthesizer model."""
        try:
            # Check if handwriting_synthesis is installed or on PYTHONPATH
            import handwriting_synthesis  # type: ignore
            from handwriting_synthesis import Hand  # type: ignore

            # Verify checkpoints
            if not os.path.exists(self.checkpoints_dir):
                logger.info(
                    "Checkpoints directory '%s' not found. "
                    "Clone https://github.com/sjvasquez/handwriting-synthesis "
                    "and download pretrained weights to use RNN mode.",
                    self.checkpoints_dir
                )
                return

            self.hand = Hand()
            logger.info("Successfully loaded sjvasquez/handwriting-synthesis RNN engine.")
        except ImportError:
            logger.debug(
                "Module 'handwriting_synthesis' not installed. "
                "Defaulting to Hershey single-stroke centerline engine."
            )
        except Exception as e:
            logger.warning("Failed to initialize RNN handwriting engine: %s", e)

    def synthesize_line(self,
                        text: str,
                        origin_x: float,
                        baseline_y: float,
                        target_cap_height_mm: float = 4.8,
                        style: int = 1,
                        bias: float = 0.85) -> Tuple[List[Stroke], float]:
        """
        Synthesizes a single line of text using the RNN and scales the coordinates
        to physical millimeters on the ruled sheet.
        
        Returns:
            (strokes_in_mm, total_advance_width_mm)
        """
        if not self.is_available():
            raise RuntimeError(
                "RNN engine is unavailable. Please install handwriting-synthesis "
                "or use the fallback Hershey vector engine."
            )

        # Generate raw strokes from RNN
        # hand.write returns strokes as list of arrays of (x, y) coordinates
        raw_strokes = self.hand.write(  # type: ignore
            lines=[text],
            biases=[bias],
            styles=[style]
        )

        if not raw_strokes or not raw_strokes[0]:
            return [], 0.0

        line_strokes = raw_strokes[0]

        # Calculate bounding box to normalize coordinates
        min_x = min(min(p[0] for p in s) for s in line_strokes if len(s) > 0)
        max_x = max(max(p[0] for p in s) for s in line_strokes if len(s) > 0)
        min_y = min(min(p[1] for p in s) for s in line_strokes if len(s) > 0)
        max_y = max(max(p[1] for p in s) for s in line_strokes if len(s) > 0)

        raw_height = max(1e-3, max_y - min_y)
        # Normalize and scale to target physical millimeters
        scale = target_cap_height_mm / raw_height

        scaled_strokes: List[Stroke] = []
        for s in line_strokes:
            stroke_pts: Stroke = []
            for px, py in s:
                x_mm = origin_x + (px - min_x) * scale
                # Align baseline: RNN max_y aligns to baseline_y
                y_mm = baseline_y - (max_y - py) * scale
                stroke_pts.append((x_mm, y_mm))
            scaled_strokes.append(stroke_pts)

        total_width = (max_x - min_x) * scale
        return scaled_strokes, total_width

