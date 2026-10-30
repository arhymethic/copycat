"""
rnn_engine.py - Exact Neural Integration for sjvasquez/handwriting-synthesis.

Repository:
https://github.com/sjvasquez/handwriting-synthesis

This module integrates the exact logic and model weights from the cloned
handwriting-synthesis repository:
1. Native Neural Network Inference (Pure NumPy implementation of Alex Graves' LSTM):
   Restores exact pre-trained model weights from checkpoints/model-17900.data-00000-of-00001
   without requiring legacy TensorFlow 1.x installations. Runs on Python 3.9 - 3.14+!
2. IAM Handwriting Style Sequences:
   Integrates the 13 human handwriting recordings from styles/style-{0..12}-strokes.npy
   and style-{0..12}-chars.npy for author style-priming.
3. Exact Stroke Processing Pipeline:
   Directly executes drawing.py algorithms:
   offsets_to_coords -> denoise (Savitzky-Golay) -> align (linear regression baseline).
4. Physical Scaling:
   Inverts vertical coordinate space and maps normalized neural strokes to 8.0 mm notebook ruled lines.
"""

import math
import os
import sys
import time
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# Type aliases
Stroke = List[Tuple[float, float]]

logger = logging.getLogger("copycat.rnn")

# Path to the cloned handwriting-synthesis repository
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handwriting-synthesis")
CHECKPOINTS_DIR = os.path.join(REPO_DIR, "checkpoints")
STYLES_DIR = os.path.join(REPO_DIR, "styles")
WEIGHTS_FILE = os.path.join(CHECKPOINTS_DIR, "model-17900.data-00000-of-00001")

# Add cloned repo to Python search path
if os.path.exists(REPO_DIR) and REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Try importing drawing from cloned repository
try:
    import drawing  # type: ignore
    DRAWING_AVAILABLE = True
except Exception as e:
    DRAWING_AVAILABLE = False
    logger.debug("Could not import drawing from handwriting-synthesis: %s", e)


# Metadata for the 13 real human styles in handwriting-synthesis/styles/
IAM_STYLE_METADATA: Dict[int, Dict[str, str]] = {
    0: {"name": "IAM Style 0", "sample": "thought that vengeance", "desc": "Cursive/print mix with wide character spacing."},
    1: {"name": "IAM Style 1", "sample": "So says the Times", "desc": "Neat small cursive with disciplined baselines."},
    2: {"name": "IAM Style 2", "sample": "House was crowded.", "desc": "Fast connected cursive with continuous word loops and steep slant."},
    3: {"name": "IAM Style 3", "sample": "A stronger finish", "desc": "Sharp angular print with upright ascenders."},
    4: {"name": "IAM Style 4", "sample": "now eased away", "desc": "Casual loose script with relaxed descenders."},
    5: {"name": "IAM Style 5", "sample": "Aurelius Antonius about A.D", "desc": "Upright formal print with deliberate strokes."},
    6: {"name": "IAM Style 6", "sample": "Byron, had said", "desc": "Upright compact cursive with narrow kerning."},
    7: {"name": "IAM Style 7", "sample": "I do not know of", "desc": "Quick light cursive with broad letterforms."},
    8: {"name": "IAM Style 8", "sample": "Holmbridge Vicarage,", "desc": "Tall flourishing script with elongated ascenders."},
    9: {"name": "IAM Style 9", "sample": "two-twenty carried, for him,", "desc": "Disciplined compact rhythmic print with uniform slant."},
    10: {"name": "IAM Style 10", "sample": "\"Just you keep thinking", "desc": "Forward slant script with pronounced character flow."},
    11: {"name": "IAM Style 11", "sample": "Royal family so often seems to find", "desc": "School script with tall ascenders and steep slant."},
    12: {"name": "IAM Style 12", "sample": "tomorrow. How selfish of me. It", "desc": "Relaxed modern handwriting with organic waviness."}
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -20.0, 20.0)))


class GravesNumPyModel:
    """
    Pure NumPy implementation of Alex Graves' recurrent neural network handwriting synthesis model.
    Loads exact weights from model-17900 without requiring TensorFlow 1.x!
    """

    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.loaded = False
        self.primed_states_cache: Dict[int, Dict[str, Any]] = {}
        self._load_weights()
        self._load_primed_states()

    def _load_weights(self):
        """Loads weight matrices and bias vectors directly from checkpoint binary."""
        if not os.path.exists(self.weights_path):
            logger.warning("Checkpoint weights not found at %s", self.weights_path)
            return

        with open(self.weights_path, "rb") as f:
            # Attention layer
            f.seek(12)
            self.att_b = np.frombuffer(f.read(120), dtype=np.float32)
            f.seek(372)
            self.att_w = np.frombuffer(f.read(57120), dtype=np.float32).reshape(476, 30)

            # LSTM 1
            f.seek(171732)
            self.lstm1_b = np.frombuffer(f.read(6400), dtype=np.float32)
            f.seek(190932)
            self.lstm1_w = np.frombuffer(f.read(3046400), dtype=np.float32).reshape(476, 1600)

            # LSTM 2
            f.seek(9330132)
            self.lstm2_b = np.frombuffer(f.read(6400), dtype=np.float32)
            f.seek(9349332)
            self.lstm2_w = np.frombuffer(f.read(5606400), dtype=np.float32).reshape(876, 1600)

            # LSTM 3
            f.seek(26168532)
            self.lstm3_b = np.frombuffer(f.read(6400), dtype=np.float32)
            f.seek(26187732)
            self.lstm3_w = np.frombuffer(f.read(5606400), dtype=np.float32).reshape(876, 1600)

            # GMM output layer
            f.seek(43006932)
            self.gmm_b = np.frombuffer(f.read(484), dtype=np.float32)
            f.seek(43008384)
            self.gmm_w = np.frombuffer(f.read(193600), dtype=np.float32).reshape(400, 121)

        self.loaded = True

    def _load_primed_states(self):
        """Loads precomputed teacher-forcing style states from disk if available."""
        # Check bundled data/primed_states.npz first, then cloned repo styles
        project_root = os.path.dirname(os.path.abspath(__file__))
        bundled_path = os.path.join(project_root, "data", "primed_states.npz")
        repo_path = os.path.join(STYLES_DIR, "primed_states.npz")
        cache_path = bundled_path if os.path.exists(bundled_path) else repo_path

        if not os.path.exists(cache_path):
            return

        try:
            data = np.load(cache_path)
            for style_idx in range(13):
                prefix = f"style_{style_idx}_"
                if f"{prefix}h1" in data:
                    self.primed_states_cache[style_idx] = {
                        "h1": data[f"{prefix}h1"],
                        "c1": data[f"{prefix}c1"],
                        "h2": data[f"{prefix}h2"],
                        "c2": data[f"{prefix}c2"],
                        "h3": data[f"{prefix}h3"],
                        "c3": data[f"{prefix}c3"],
                        "kappa": data[f"{prefix}kappa"],
                        "w": data[f"{prefix}w"],
                        "c_prime": str(data[f"{prefix}c_prime"]),
                    }
        except Exception as e:
            logger.warning("Failed to load primed states cache: %s", e)

    @staticmethod
    def _lstm_cell(x: np.ndarray, c: np.ndarray, h: np.ndarray,
                   kernel: np.ndarray, bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TF-equivalent LSTM cell forward step."""
        inp = np.concatenate([x, h])
        gates = inp @ kernel + bias
        # Gate order: i, j, f, o
        i_g = _sigmoid(gates[0:400])
        j_g = np.tanh(gates[400:800])
        f_g = _sigmoid(gates[800:1200] + 1.0)
        o_g = _sigmoid(gates[1200:1600])
        c_new = c * f_g + i_g * j_g
        h_new = np.tanh(c_new) * o_g
        return c_new, h_new

    def sample_line(self, text: str, style_id: int = 1, bias: float = 0.85) -> np.ndarray:
        """
        Synthesizes a single line of text with style-priming from IAM style.
        Uses precomputed teacher-forcing state cache for fast generation.
        Returns array of offsets [dx, dy, eos].
        """
        if not self.loaded:
            raise RuntimeError("Graves RNN model weights are not loaded.")

        t_start = time.time()
        style_idx = style_id % 13

        # Map text to valid alphabet characters
        valid_charset = set(drawing.alphabet)
        cleaned_text = []
        for ch in text:
            if ch in valid_charset:
                cleaned_text.append(ch)
            elif ch.lower() in valid_charset:
                cleaned_text.append(ch.lower())
            elif ch == "=":
                cleaned_text.append("-")
            else:
                cleaned_text.append(" ")
        clean_line = "".join(cleaned_text)[:72]
        if not clean_line.strip():
            return np.zeros((0, 3), dtype=np.float32)

        # 1. Retrieve or compute teacher-forcing state for this style
        if style_idx in self.primed_states_cache:
            ps = self.primed_states_cache[style_idx]
            h1 = ps["h1"].copy()
            c1 = ps["c1"].copy()
            h2 = ps["h2"].copy()
            c2 = ps["c2"].copy()
            h3 = ps["h3"].copy()
            c3 = ps["c3"].copy()
            kappa = ps["kappa"].copy()
            w = ps["w"].copy()
            c_prime = ps["c_prime"]
        else:
            strokes_path = os.path.join(STYLES_DIR, f"style-{style_idx}-strokes.npy")
            chars_path = os.path.join(STYLES_DIR, f"style-{style_idx}-chars.npy")
            x_prime = np.load(strokes_path)
            c_prime = np.load(chars_path).tobytes().decode("utf-8")

            encoded_prime = drawing.encode_ascii(c_prime)
            U_prime = len(encoded_prime)
            c_oh_prime = np.zeros((U_prime, len(drawing.alphabet)), dtype=np.float32)
            for u_idx, code in enumerate(encoded_prime):
                c_oh_prime[u_idx, code] = 1.0
            u_vec_prime = np.arange(U_prime, dtype=np.float32)

            h1 = np.zeros(400, dtype=np.float32)
            c1 = np.zeros(400, dtype=np.float32)
            h2 = np.zeros(400, dtype=np.float32)
            c2 = np.zeros(400, dtype=np.float32)
            h3 = np.zeros(400, dtype=np.float32)
            c3 = np.zeros(400, dtype=np.float32)
            kappa = np.zeros(10, dtype=np.float32)
            w = np.zeros(73, dtype=np.float32)

            for t in range(len(x_prime)):
                inp = x_prime[t]
                s1_in = np.concatenate([w, inp])
                c1, h1 = self._lstm_cell(s1_in, c1, h1, self.lstm1_w, self.lstm1_b)

                att_in = np.concatenate([w, inp, h1])
                att_p = _softplus(att_in @ self.att_w + self.att_b)
                alpha = att_p[0:10]
                beta = np.maximum(att_p[10:20], 0.01)
                kappa = kappa + att_p[20:30] / 25.0

                phi = np.sum(alpha[:, None] * np.exp(-((kappa[:, None] - u_vec_prime[None, :]) ** 2) / beta[:, None]), axis=0)
                w = np.sum(phi[:, None] * c_oh_prime, axis=0)

                s2_in = np.concatenate([inp, h1, w])
                c2, h2 = self._lstm_cell(s2_in, c2, h2, self.lstm2_w, self.lstm2_b)

                s3_in = np.concatenate([inp, h2, w])
                c3, h3 = self._lstm_cell(s3_in, c3, h3, self.lstm3_w, self.lstm3_b)

            self.primed_states_cache[style_idx] = {
                "h1": h1.copy(), "c1": c1.copy(),
                "h2": h2.copy(), "c2": c2.copy(),
                "h3": h3.copy(), "c3": c3.copy(),
                "kappa": kappa.copy(), "w": w.copy(),
                "c_prime": c_prime
            }

        full_text = c_prime + " " + clean_line
        encoded = drawing.encode_ascii(full_text)
        U = len(encoded)

        # One-hot encoding for characters
        c_one_hot = np.zeros((U, len(drawing.alphabet)), dtype=np.float32)
        for u_idx, code in enumerate(encoded):
            c_one_hot[u_idx, code] = 1.0
        u_vec = np.arange(U, dtype=np.float32)

        # 2. Free-running generative phase: sample new text
        sampled: List[np.ndarray] = []
        curr_inp = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        max_steps = 38 * len(clean_line)

        for _ in range(max_steps):
            s1_in = np.concatenate([w, curr_inp])
            c1, h1 = self._lstm_cell(s1_in, c1, h1, self.lstm1_w, self.lstm1_b)

            att_in = np.concatenate([w, curr_inp, h1])
            att_p = _softplus(att_in @ self.att_w + self.att_b)
            alpha = att_p[0:10]
            beta = np.maximum(att_p[10:20], 0.01)
            kappa = kappa + att_p[20:30] / 25.0

            phi = np.sum(alpha[:, None] * np.exp(-((kappa[:, None] - u_vec[None, :]) ** 2) / beta[:, None]), axis=0)
            w = np.sum(phi[:, None] * c_one_hot, axis=0)

            s2_in = np.concatenate([curr_inp, h1, w])
            c2, h2 = self._lstm_cell(s2_in, c2, h2, self.lstm2_w, self.lstm2_b)

            s3_in = np.concatenate([curr_inp, h2, w])
            c3, h3 = self._lstm_cell(s3_in, c3, h3, self.lstm3_w, self.lstm3_b)

            # GMM parameters
            gmm_p = h3 @ self.gmm_w + self.gmm_b
            pis = gmm_p[0:20] * (1.0 + bias)
            sigmas = gmm_p[20:60] - bias
            rhos = gmm_p[60:80]
            mus = gmm_p[80:120]
            e = gmm_p[120]

            # Softmax for mixture weights
            exp_pis = np.exp(pis - np.max(pis))
            pis = exp_pis / np.sum(exp_pis)
            pis = np.where(pis < 0.01, 0.0, pis)
            pis = pis / np.sum(pis)

            sigmas = np.clip(np.exp(sigmas), 1e-4, 1e5)
            sigma1 = sigmas[0:20]
            sigma2 = sigmas[20:40]
            rhos = np.clip(np.tanh(rhos), -0.99999, 0.99999)
            mu1 = mus[0:20]
            mu2 = mus[20:40]

            eos_prob = np.clip(_sigmoid(e), 1e-8, 1.0 - 1e-8)
            eos_prob = 0.0 if eos_prob < 0.01 else eos_prob

            # Sample component
            k = np.random.choice(20, p=pis)
            mean = [mu1[k], mu2[k]]
            cov = [[sigma1[k] ** 2, rhos[k] * sigma1[k] * sigma2[k]],
                   [rhos[k] * sigma1[k] * sigma2[k], sigma2[k] ** 2]]
            dx, dy = np.random.multivariate_normal(mean, cov)
            eos = 1.0 if np.random.rand() < eos_prob else 0.0

            curr_inp = np.array([dx, dy, eos], dtype=np.float32)
            sampled.append(curr_inp)

            # Check termination
            char_idx = np.argmax(phi)
            if char_idx >= U - 1 and eos == 1.0:
                break

        elapsed = time.time() - t_start
        print(f"      [RNN] Synthesized line in {elapsed:.2f}s ({len(sampled)} points, style={style_idx}, bias={bias:.2f})")
        return np.array(sampled)


def transform_strokes(strokes: List[Stroke],
                      baseline_y: float,
                      origin_x: float,
                      slant_deg: float = 0.0,
                      scale_mult: float = 1.0,
                      jitter_mm: float = 0.0,
                      wobble: float = 0.0) -> List[Stroke]:
    """Applies slant shear, scale, jitter, and wobble transforms to stroke polylines in physical mm."""
    if not strokes:
        return []

    tan_slant = math.tan(math.radians(-slant_deg)) if slant_deg != 0.0 else 0.0
    transformed: List[Stroke] = []

    for s_idx, stroke in enumerate(strokes):
        new_stroke: Stroke = []
        stroke_jitter_y = (np.random.normal(0.0, jitter_mm) if jitter_mm > 0.0 else 0.0)

        for pt_idx, (x, y) in enumerate(stroke):
            dy = (y - baseline_y) * scale_mult
            dx = (x - origin_x) * scale_mult

            if tan_slant != 0.0:
                dx += dy * tan_slant

            if wobble > 0.0:
                w_dx = wobble * 0.4 * math.sin(0.8 * pt_idx + s_idx)
                w_dy = wobble * 0.4 * math.cos(0.8 * pt_idx + s_idx)
                dx += w_dx
                dy += w_dy

            new_stroke.append((float(origin_x + dx), float(baseline_y + dy + stroke_jitter_y)))

        transformed.append(new_stroke)

    return transformed


class VasquezHandwritingEngine:
    """
    Direct bridge to sjvasquez/handwriting-synthesis.
    Provides fast, authentic Graves LSTM handwriting inference with caching.
    """

    def __init__(self):
        self.repo_dir = REPO_DIR
        self.checkpoints_dir = CHECKPOINTS_DIR
        self.styles_dir = STYLES_DIR
        self.weights_file = WEIGHTS_FILE

        self.has_repo = os.path.exists(self.repo_dir)
        self.has_checkpoints = os.path.exists(self.weights_file)
        self.has_styles = os.path.exists(self.styles_dir)

        self.neural_model: Optional[GravesNumPyModel] = None
        self._raw_offsets_cache: Dict[Tuple[str, int, float], np.ndarray] = {}
        self._line_cache: Dict[Tuple[str, int, float], Tuple[List[Stroke], float]] = {}
        self._init_model()

    def _init_model(self):
        """Initializes the pure NumPy Graves LSTM model from pre-trained weights."""
        if self.has_checkpoints:
            try:
                self.neural_model = GravesNumPyModel(self.weights_file)
                logger.info("Loaded Graves RNN model weights into pure NumPy engine.")
            except Exception as e:
                logger.warning("Could not initialize NumPy neural model: %s", e)
                self.neural_model = None

    def is_rnn_ready(self) -> bool:
        """Returns True if the Graves RNN weights are loaded and ready to infer."""
        return self.neural_model is not None and self.neural_model.loaded

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
                        target_cap_height_mm: float = 4.8,
                        slant_deg: float = 0.0,
                        jitter_mm: float = 0.0,
                        wobble: float = 0.0) -> Tuple[List[Stroke], float]:
        """
        Synthesizes a line of text using the exact handwriting-synthesis logic:
        1. Alex Graves RNN generation with style priming (cached in memory)
        2. Applies drawing.offsets_to_coords, drawing.denoise, and drawing.align
        3. Scales coordinates to physical millimeters locked to 8.0 mm baselines
        4. Applies interactive fine-tuning parameters (slant, jitter, wobble).
        """
        if not self.is_rnn_ready():
            raise RuntimeError(
                f"Graves RNN model weights are missing or not loaded from {self.weights_file}."
            )

        cache_key = (text.strip(), style_id % 13, round(bias, 2))

        # Check raw offsets cache to avoid slow re-sampling when adjusting sliders
        if cache_key in self._raw_offsets_cache:
            raw_offsets = self._raw_offsets_cache[cache_key]
        else:
            t0 = time.time()
            raw_offsets = self.neural_model.sample_line(text, style_id=style_id, bias=bias)  # type: ignore
            if len(raw_offsets) == 0:
                return [], 0.0
            self._raw_offsets_cache[cache_key] = raw_offsets
            preview = text[:28] + ("..." if len(text) > 28 else "")
            print(f"      [SYNTH] Line \"{preview}\" sampled ({len(raw_offsets)} points, {time.time() - t0:.2f}s)")

        # Convert raw offsets to physical millimeters
        polylines, total_width = self._process_offsets_to_mm(
            raw_offsets, origin_x=origin_x, baseline_y=baseline_y, target_cap_height_mm=target_cap_height_mm
        )

        # Apply interactive fine-tuning parameters (slant shear, micro-jitter, tremor)
        if slant_deg != 0.0 or jitter_mm > 0.0 or wobble > 0.0:
            polylines = transform_strokes(
                polylines,
                baseline_y=baseline_y,
                origin_x=origin_x,
                slant_deg=slant_deg,
                scale_mult=1.0,
                jitter_mm=jitter_mm,
                wobble=wobble
            )

        return polylines, total_width

    def _process_offsets_to_mm(self,
                               offsets: np.ndarray,
                               origin_x: float,
                               baseline_y: float,
                               target_cap_height_mm: float) -> Tuple[List[Stroke], float]:
        """
        Applies the exact drawing.py pipeline from sjvasquez/handwriting-synthesis
        and maps coordinates to physical millimeters on the notebook paper.
        """
        offsets = offsets.copy()

        # 1. Strip priming residual drift before the first pen-up jump to the text
        eos_indices = np.where(offsets[:, 2] == 1.0)[0]
        if len(eos_indices) > 0 and eos_indices[0] < 6:
            # Skip the residual priming points up to and including the first eos=1.0
            offsets = offsets[eos_indices[0] + 1:]

        if len(offsets) == 0:
            return [], 0.0

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
