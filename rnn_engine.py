"""
rnn_engine.py - Exact Integration for sjvasquez/handwriting-synthesis.

Repository:
https://github.com/sjvasquez/handwriting-synthesis

This module integrates the exact logic and assets from the cloned
handwriting-synthesis repository:
1. Neural Network Inference: Uses Alex Graves' RNN via Hand from demo.py
   when TensorFlow is installed, restoring weights from checkpoints/model-17900.
2. IAM Handwriting Style Sequences: Integrates the 13 human handwriting recordings
   from styles/style-{0..12}-strokes.npy and style-{0..12}-chars.npy.
3. Exact Stroke Processing: Integrates drawing.py algorithms (offsets_to_coords,
   align, denoise, interpolate, and skew).
4. Physical Scaling: Maps raw coordinate strokes to 8.0 mm notebook ruled lines.
"""

import os
import sys
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np

# Type aliases
Stroke = List[Tuple[float, float]]

logger = logging.getLogger("copycat.rnn")

# Path to the cloned handwriting-synthesis repository
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handwriting-synthesis")
CHECKPOINTS_DIR = os.path.join(REPO_DIR, "checkpoints")
STYLES_DIR = os.path.join(REPO_DIR, "styles")

# Add cloned repo to Python search path so its modules (drawing, rnn, etc.) are accessible
if os.path.exists(REPO_DIR) and REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Try importing drawing from cloned repository
try:
    import drawing  # type: ignore
    DRAWING_AVAILABLE = True
except Exception as e:
    DRAWING_AVAILABLE = False
    logger.debug("Could not import drawing from handwriting-synthesis: %s", e)

# Try importing Hand from cloned repository (requires tensorflow)
TF_AVAILABLE = False
HAND_CLASS = None
try:
    import tensorflow as tf  # type: ignore
    from demo import Hand  # type: ignore
    TF_AVAILABLE = True
    HAND_CLASS = Hand
except Exception as e:
    TF_AVAILABLE = False
    logger.debug("TensorFlow/Hand could not be loaded: %s", e)


# Metadata for the 13 real human styles in handwriting-synthesis/styles/
IAM_STYLE_METADATA: Dict[int, Dict[str, str]] = {
    0: {"name": "IAM Style 0", "sample": "thought that vengeance", "desc": "Cursive/print mix with wide character spacing."},
    1: {"name": "IAM Style 1", "sample": "So says the Times", "desc": "Neat small cursive with disciplined baselines."},
    2: {"name": "IAM Style 2", "sample": "House was crowded.", "desc": "Fast connected cursive with continuous word loops."},
    3: {"name": "IAM Style 3", "sample": "A stronger finish", "desc": "Sharp angular print with upright ascenders."},
    4: {"name": "IAM Style 4", "sample": "now eased away", "desc": "Casual loose script with relaxed descenders."},
    5: {"name": "IAM Style 5", "sample": "Aurelius Antonius about A.D", "desc": "Upright formal print with deliberate strokes."},
    6: {"name": "IAM Style 6", "sample": "Byron, had said", "desc": "Upright compact cursive with narrow kerning."},
    7: {"name": "IAM Style 7", "sample": "I do not know of", "desc": "Quick light cursive with delicate pen pressure."},
    8: {"name": "IAM Style 8", "sample": "Holmbridge Vicarage,", "desc": "Tall flourishing script with elongated ascenders."},
    9: {"name": "IAM Style 9", "sample": "two-twenty carried, for him,", "desc": "Disciplined rhythmic print with uniform slant."},
    10: {"name": "IAM Style 10", "sample": "\"Just you keep thinking", "desc": "Forward slant script with pronounced character flow."},
    11: {"name": "IAM Style 11", "sample": "Royal family so often seems to find", "desc": "Rounded school script with balanced proportions."},
    12: {"name": "IAM Style 12", "sample": "tomorrow. How selfish of me. It", "desc": "Relaxed modern handwriting with organic waviness."}
}


class VasquezHandwritingEngine:
    """
    Direct bridge to sjvasquez/handwriting-synthesis.
    """

    def __init__(self):
        self.repo_dir = REPO_DIR
        self.checkpoints_dir = CHECKPOINTS_DIR
        self.styles_dir = STYLES_DIR
        self.has_repo = os.path.exists(self.repo_dir)
        self.has_checkpoints = os.path.exists(os.path.join(CHECKPOINTS_DIR, "checkpoint"))
        self.has_styles = os.path.exists(self.styles_dir)
        self.hand_instance = None
        self._init_hand()

    def _init_hand(self):
        """Initializes Hand() from demo.py if TensorFlow is present."""
        if TF_AVAILABLE and HAND_CLASS is not None and self.has_checkpoints:
            try:
                # Save working directory and switch into repo dir so relative paths in Hand work
                cur_dir = os.getcwd()
                os.chdir(self.repo_dir)
                self.hand_instance = HAND_CLASS()
                os.chdir(cur_dir)
                logger.info("Loaded Graves RNN model from checkpoints/model-17900.")
            except Exception as e:
                logger.warning("Could not initialize Hand model: %s", e)
                self.hand_instance = None

    def is_rnn_ready(self) -> bool:
        """Returns True if TensorFlow and the Graves RNN model weights are loaded."""
        return self.hand_instance is not None

    def get_style_info(self, style_id: int) -> Dict[str, str]:
        """Returns description and priming text for a style (0-12)."""
        return IAM_STYLE_METADATA.get(style_id % 13, IAM_STYLE_METADATA[0])

    def load_iam_style_raw(self, style_id: int) -> Tuple[np.ndarray, str]:
        """Loads raw recorded strokes and characters for an IAM style from repo."""
        style_idx = style_id % 13
        strokes_path = os.path.join(self.styles_dir, f"style-{style_idx}-strokes.npy")
        chars_path = os.path.join(self.styles_dir, f"style-{style_idx}-chars.npy")
        if not os.path.exists(strokes_path) or not os.path.exists(chars_path):
            raise FileNotFoundError(f"Style files not found for style {style_idx} in {self.styles_dir}")

        strokes = np.load(strokes_path)
        chars = np.load(chars_path).tobytes().decode("utf-8")
        return strokes, chars

    def synthesize_line(self,
                        text: str,
                        origin_x: float,
                        baseline_y: float,
                        style_id: int = 1,
                        bias: float = 0.85,
                        target_cap_height_mm: float = 4.8) -> Tuple[List[Stroke], float]:
        """
        Synthesizes a line of text using the exact handwriting-synthesis logic:
        1. Neural sampling using Hand (if TF is available)
        2. Applies drawing.offsets_to_coords, drawing.denoise, and drawing.align
        3. Scales coordinates to physical millimeters locked to 8.0 mm baselines.
        """
        if not self.is_rnn_ready():
            raise RuntimeError(
                f"TensorFlow is required to run neural RNN synthesis with sjvasquez/handwriting-synthesis.\n"
                f"Cloned repository is present at: {self.repo_dir}\n"
                f"Checkpoints (42 MB) are present at: {self.checkpoints_dir}\n"
                f"To enable neural inference, install TensorFlow in your environment:\n"
                f"    pip install tensorflow==1.15.0  (or tensorflow-cpu)\n"
            )

        # Truncate text to 75 characters max as required by the model
        text_line = text[:74]

        # Switch to repo dir for style loading inside Hand
        cur_dir = os.getcwd()
        try:
            os.chdir(self.repo_dir)
            # Sample strokes: returns list of numpy arrays of [dx, dy, eos]
            raw_samples = self.hand_instance._sample(
                lines=[text_line],
                biases=[bias],
                styles=[style_id % 13]
            )
        finally:
            os.chdir(cur_dir)

        if not raw_samples or len(raw_samples[0]) == 0:
            return [], 0.0

        offsets = raw_samples[0].copy()
        return self._process_offsets_to_mm(offsets, origin_x, baseline_y, target_cap_height_mm)

    def _process_offsets_to_mm(self,
                               offsets: np.ndarray,
                               origin_x: float,
                               baseline_y: float,
                               target_cap_height_mm: float) -> Tuple[List[Stroke], float]:
        """
        Applies the exact drawing.py pipeline from sjvasquez/handwriting-synthesis
        and maps coordinates to physical millimeters on the notebook paper.
        """
        offsets[:, :2] *= 1.5

        if DRAWING_AVAILABLE:
            coords = drawing.offsets_to_coords(offsets)
            coords = drawing.denoise(coords)
            coords[:, :2] = drawing.align(coords[:, :2])
        else:
            coords = np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)

        # Align coordinate orientation: flip Y so positive Y is downward
        coords[:, 1] *= -1.0

        # Normalization
        min_x = coords[:, 0].min()
        max_x = coords[:, 0].max()
        min_y = coords[:, 1].min()
        max_y = coords[:, 1].max()

        raw_height = max(1e-3, max_y - min_y)
        scale = target_cap_height_mm / raw_height

        # Convert to polylines separated by end-of-stroke (eos == 1.0)
        polylines: List[Stroke] = []
        current_stroke: Stroke = []

        for x, y, eos in coords:
            # Map to page mm
            px = origin_x + (x - min_x) * scale
            py = baseline_y - (max_y - y) * scale
            current_stroke.append((float(px), float(py)))

            if eos == 1.0:
                if len(current_stroke) >= 2:
                    polylines.append(current_stroke)
                current_stroke = []

        if current_stroke and len(current_stroke) >= 2:
            polylines.append(current_stroke)

        total_width = (max_x - min_x) * scale
        return polylines, total_width


# Global instance
VASQUEZ_ENGINE = VasquezHandwritingEngine()
