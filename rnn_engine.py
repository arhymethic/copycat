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
import ctypes
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import stroke_font

# Type aliases
Stroke = List[Tuple[float, float]]

logger = logging.getLogger("copycat.rnn")

# Project root and asset directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPO_DIR = os.path.join(PROJECT_ROOT, "handwriting-synthesis")

# Weights: Prefer compressed data/weights.npz (12.8 MB), fallback to legacy checkpoint (43.5 MB)
WEIGHTS_NPZ = os.path.join(DATA_DIR, "weights.npz")
LEGACY_WEIGHTS = os.path.join(REPO_DIR, "checkpoints", "model-17900.data-00000-of-00001")
WEIGHTS_FILE = WEIGHTS_NPZ if os.path.exists(WEIGHTS_NPZ) else LEGACY_WEIGHTS
CHECKPOINTS_DIR = os.path.dirname(WEIGHTS_FILE)

# Styles: Prefer data/styles, fallback to handwriting-synthesis/styles
DATA_STYLES = os.path.join(DATA_DIR, "styles")
REPO_STYLES = os.path.join(REPO_DIR, "styles")
STYLES_DIR = DATA_STYLES if os.path.exists(DATA_STYLES) else REPO_STYLES

# Add project root and repo to Python search path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.exists(REPO_DIR) and REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Try importing drawing module (local drawing.py or from cloned repo)
try:
    import drawing  # type: ignore
    DRAWING_AVAILABLE = True
except Exception as e:
    DRAWING_AVAILABLE = False
    logger.debug("Could not import drawing: %s", e)


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
        self._c_lib = self._init_c_accelerator()
        self._load_weights()
        self._load_primed_states()

    def _init_c_accelerator(self):
        """Attempts to load or compile the native C-AVX2 accelerator library for 10x-20x speedup."""
        so_path = os.path.join(DATA_DIR, "librnn_fast.so")
        c_path = os.path.join(DATA_DIR, "fast_graves.c")

        if not os.path.exists(so_path) and os.path.exists(c_path):
            try:
                import subprocess
                subprocess.run(
                    ["gcc", "-O3", "-mavx2", "-mfma", "-shared", "-fPIC", c_path, "-o", so_path, "-lm"],
                    check=True,
                    capture_output=True
                )
            except Exception as e:
                logger.debug("Could not compile C accelerator: %s", e)
                return None

        if os.path.exists(so_path):
            try:
                lib = ctypes.CDLL(so_path)
                lib.fast_sample_line.argtypes = [
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                    ctypes.c_float, ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint
                ]
                lib.fast_sample_line.restype = ctypes.c_int
                logger.info("Loaded native C-AVX2 neural accelerator.")
                return lib
            except Exception as e:
                logger.debug("Failed loading librnn_fast.so: %s", e)
                return None
        return None

    def _load_weights(self):
        """Loads weight matrices and bias vectors from compressed npz or raw binary."""
        if not os.path.exists(self.weights_path):
            logger.warning("Checkpoint weights not found at %s", self.weights_path)
            return

        if self.weights_path.endswith(".npz"):
            data = np.load(self.weights_path)
            self.att_b = data["att_b"]
            self.att_w = data["att_w"]
            self.lstm1_b = data["lstm1_b"]
            self.lstm1_w = data["lstm1_w"]
            self.lstm2_b = data["lstm2_b"]
            self.lstm2_w = data["lstm2_w"]
            self.lstm3_b = data["lstm3_b"]
            self.lstm3_w = data["lstm3_w"]
            self.gmm_b = data["gmm_b"]
            self.gmm_w = data["gmm_w"]
            self.loaded = True
            logger.info("Loaded optimized compressed neural weights from %s", self.weights_path)
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
            else:
                cleaned_text.append(" ")
        raw_clean = "".join(cleaned_text)[:96]
        # Collapse multiple spaces into single space so Graves attention window never slips
        clean_line = " ".join(raw_clean.split())
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
        max_steps = 38 * len(clean_line)

        # Fast path: Native C-AVX2 compiled kernel (10x-20x speedup)
        if self._c_lib is not None:
            out_buf = np.zeros((max_steps, 3), dtype=np.float32)
            seed = int(time.time() * 1000) & 0x7FFFFFFF

            steps_sampled = self._c_lib.fast_sample_line(
                self.att_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.att_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.lstm1_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.lstm1_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.lstm2_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.lstm2_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.lstm3_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.lstm3_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.gmm_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.gmm_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                h1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                h2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                h3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c_one_hot.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                U,
                ctypes.c_float(bias),
                max_steps,
                out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                seed
            )
            sampled = out_buf[:steps_sampled]
            elapsed = time.time() - t_start
            print(f"      [RNN-FAST] Synthesized line in {elapsed:.2f}s ({steps_sampled} points, style={style_idx}, bias={bias:.2f}) [C-AVX2]")
            return sampled

        # Fallback path: Pure NumPy loop
        sampled: List[np.ndarray] = []
        curr_inp = np.array([0.0, 0.0, 0.0], dtype=np.float32)

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


class DiskLineCache:
    """Persistent SQLite-backed cache for sampled neural stroke offsets."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                try:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS line_cache (
                            key TEXT PRIMARY KEY,
                            n_points INTEGER,
                            data BLOB
                        )
                    """)
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            logger.debug("Failed to init SQLite line cache: %s", e)

    def get(self, key: str) -> Optional[np.ndarray]:
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=3.0)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT n_points, data FROM line_cache WHERE key = ?", (key,))
                    row = cur.fetchone()
                    if row:
                        n_points, blob = row
                        return np.frombuffer(blob, dtype=np.float32).reshape(n_points, 3).copy()
                finally:
                    conn.close()
        except Exception as e:
            logger.debug("Disk cache get failed: %s", e)
        return None

    def put(self, key: str, offsets: np.ndarray):
        try:
            n_points = len(offsets)
            blob = offsets.astype(np.float32).tobytes()
            with self._lock:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO line_cache (key, n_points, data) VALUES (?, ?, ?)",
                        (key, n_points, blob)
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            logger.debug("Disk cache put failed: %s", e)


MATH_SYMBOLS = set("=+*/^_<>[]{}|\\$%&~")


def synthesize_hybrid_line(text: str,
                           origin_x: float,
                           baseline_y: float,
                           model_engine,
                           style_id: int = 1,
                           bias: float = 0.85,
                           target_cap_height_mm: float = 3.8,
                           slant_deg: float = 0.0,
                           jitter_mm: float = 0.0,
                           wobble: float = 0.0,
                           letter_spacing: float = 1.0,
                           space_mult: float = 1.0) -> Tuple[List[Stroke], float]:
    """
    Synthesizes a line containing mathematical symbols, operators, equations, or
    subscripts/superscripts while preserving neural RNN handwriting for all words
    and characters. Only missing math operators (=, +, *, /, ^, _) are rendered
    with inline procedural single-stroke glyphs.
    """
    val_chars = set(drawing.alphabet) - {'\x00'}

    tokens = []
    curr_text = []
    for ch in text:
        if ch in val_chars:
            curr_text.append(ch)
        else:
            if curr_text:
                tokens.append(('rnn', ''.join(curr_text)))
                curr_text = []
            tokens.append(('sym', ch))
    if curr_text:
        tokens.append(('rnn', ''.join(curr_text)))

    all_strokes: List[Stroke] = []
    curr_x = origin_x
    scale_sym = target_cap_height_mm / 14.0
    base_space_w = 2.4 * (target_cap_height_mm / 3.8) * space_mult

    idx = 0
    n = len(tokens)
    while idx < n:
        kind, val = tokens[idx]

        # 1. Superscript operator (e.g. ^2, ^x)
        if kind == 'sym' and val == '^' and idx + 1 < n:
            next_kind, next_val = tokens[idx + 1]
            sup_y = baseline_y - 2.0 * (target_cap_height_mm / 3.8)
            sup_scale = target_cap_height_mm * 0.72

            if next_kind == 'sym':
                s_strokes, adv = stroke_font.render_word_strokes(
                    next_val, curr_x, sup_y, sup_scale / 14.0,
                    slant_deg if slant_deg != 0.0 else -5.0,
                    max(0.04, jitter_mm), 0.0, max(0.04, wobble)
                )
                all_strokes.extend(s_strokes)
                curr_x += adv
            else:
                core = next_val.strip()
                if core:
                    raw_offsets = model_engine.get_or_sample_offsets(core, style_id=style_id, bias=bias)
                    polys, width = model_engine._process_offsets_to_mm(
                        raw_offsets, origin_x=curr_x, baseline_y=sup_y,
                        target_cap_height_mm=sup_scale, letter_spacing=letter_spacing
                    )
                    all_strokes.extend(polys)
                    curr_x += width
            idx += 2
            continue

        # 2. Subscript operator (e.g. _f, _i, _1)
        if kind == 'sym' and val == '_' and idx + 1 < n:
            next_kind, next_val = tokens[idx + 1]
            sub_y = baseline_y + 1.2 * (target_cap_height_mm / 3.8)
            sub_scale = target_cap_height_mm * 0.72

            if next_kind == 'sym':
                s_strokes, adv = stroke_font.render_word_strokes(
                    next_val, curr_x, sub_y, sub_scale / 14.0,
                    slant_deg if slant_deg != 0.0 else -5.0,
                    max(0.04, jitter_mm), 0.0, max(0.04, wobble)
                )
                all_strokes.extend(s_strokes)
                curr_x += adv
            else:
                core = next_val.strip()
                if core:
                    raw_offsets = model_engine.get_or_sample_offsets(core, style_id=style_id, bias=bias)
                    polys, width = model_engine._process_offsets_to_mm(
                        raw_offsets, origin_x=curr_x, baseline_y=sub_y,
                        target_cap_height_mm=sub_scale, letter_spacing=letter_spacing
                    )
                    all_strokes.extend(polys)
                    curr_x += width
            idx += 2
            continue

        # 3. Procedural symbol (=, +, *, /, etc.)
        if kind == 'sym':
            s_strokes, adv = stroke_font.render_word_strokes(
                val, curr_x, baseline_y, scale_sym,
                slant_deg if slant_deg != 0.0 else -5.0,
                max(0.04, jitter_mm), 0.0, max(0.04, wobble)
            )
            all_strokes.extend(s_strokes)
            curr_x += adv
        else:
            # 4. Neural RNN text chunk
            l_spaces = len(val) - len(val.lstrip(' '))
            r_spaces = len(val) - len(val.rstrip(' '))
            core = val.strip()

            curr_x += l_spaces * base_space_w
            if core:
                raw_offsets = model_engine.get_or_sample_offsets(core, style_id=style_id, bias=bias)
                polys, width = model_engine._process_offsets_to_mm(
                    raw_offsets, origin_x=curr_x, baseline_y=baseline_y,
                    target_cap_height_mm=target_cap_height_mm, letter_spacing=letter_spacing
                )
                all_strokes.extend(polys)
                curr_x += width
            curr_x += r_spaces * base_space_w

        idx += 1

    return all_strokes, curr_x - origin_x


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

        self.has_repo = os.path.exists(self.repo_dir) or os.path.exists(DATA_DIR)
        self.has_checkpoints = os.path.exists(self.weights_file)
        self.has_styles = os.path.exists(self.styles_dir)

        self.neural_model: Optional[GravesNumPyModel] = None
        self._cache_lock = threading.Lock()
        self._raw_offsets_cache: Dict[Tuple[str, int, float], np.ndarray] = {}
        self._line_cache: Dict[Tuple[str, int, float], Tuple[List[Stroke], float]] = {}

        db_path = os.path.join(DATA_DIR, "cache", "neural_cache.db")
        self.disk_cache = DiskLineCache(db_path)

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

    def is_cached(self, text: str, style_id: int = 1, bias: float = 0.85) -> bool:
        """Returns True if the line's raw offsets are already cached in RAM or on disk."""
        clean_text = text.strip()
        cache_key = (clean_text, style_id % 13, round(bias, 2))
        with self._cache_lock:
            if cache_key in self._raw_offsets_cache:
                return True
        key_str = f"{clean_text}__s{style_id % 13}__b{round(bias, 2)}"
        return self.disk_cache.get(key_str) is not None

    def get_or_sample_offsets(self, text: str, style_id: int = 1, bias: float = 0.85) -> np.ndarray:
        """Fetches raw offsets from RAM/disk cache or synthesizes via neural model."""
        if not self.is_rnn_ready():
            raise RuntimeError(
                f"Graves RNN model weights are missing or not loaded from {self.weights_file}."
            )

        clean_text = text.strip()
        cache_key = (clean_text, style_id % 13, round(bias, 2))
        with self._cache_lock:
            if cache_key in self._raw_offsets_cache:
                return self._raw_offsets_cache[cache_key]

        key_str = f"{clean_text}__s{style_id % 13}__b{round(bias, 2)}"
        disk_val = self.disk_cache.get(key_str)
        if disk_val is not None:
            with self._cache_lock:
                self._raw_offsets_cache[cache_key] = disk_val
            return disk_val

        t0 = time.time()
        raw_offsets = self.neural_model.sample_line(text, style_id=style_id, bias=bias)  # type: ignore
        if len(raw_offsets) > 0:
            with self._cache_lock:
                self._raw_offsets_cache[cache_key] = raw_offsets
            self.disk_cache.put(key_str, raw_offsets)
            preview = text[:28] + ("..." if len(text) > 28 else "")
            print(f"      [SYNTH] Line \"{preview}\" sampled ({len(raw_offsets)} points, {time.time() - t0:.2f}s)")
        return raw_offsets

    def prefetch_lines(self, texts: List[str], style_id: int = 1, bias: float = 0.85):
        """Synthesizes multiple lines in parallel across CPU cores using a ThreadPoolExecutor."""
        # Math formulas render procedurally; only prefetch pure text lines
        rnn_texts = [t for t in texts if not any(ch in MATH_SYMBOLS for ch in t)]
        uncached = [t for t in rnn_texts if not self.is_cached(t, style_id, bias)]
        if not uncached:
            return

        workers = min(4, len(uncached), os.cpu_count() or 4)
        if workers <= 1:
            for t in uncached:
                self.get_or_sample_offsets(t, style_id, bias)
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self.get_or_sample_offsets, t, style_id, bias)
                for t in uncached
            ]
            for f in futures:
                try:
                    f.result()
                except Exception as e:
                    logger.warning("Parallel line synthesis error: %s", e)

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
                        target_cap_height_mm: float = 3.8,
                        slant_deg: float = 0.0,
                        jitter_mm: float = 0.0,
                        wobble: float = 0.0,
                        letter_spacing: float = 1.0,
                        space_mult: float = 1.0) -> Tuple[List[Stroke], float]:
        """
        Synthesizes a line of text:
        1. Lines containing mathematical operators (=, +, *, /, ^, _) use hybrid synthesis:
           all words, variables, and numbers use authentic neural RNN handwriting in the active
           author style, while missing math operators are drawn inline with single-stroke glyphs.
        2. Question headers starting with capital 'Q' (e.g. Q1., Q2.) render 'Q' with vector typography
           and the remainder via Alex Graves neural RNN.
        3. Regular sentences use continuous neural handwriting, with word-spacing scaled by space_mult.
        """
        # 1. Math formulas and equations: hybrid synthesis with neural variables & vector operators
        if any(ch in MATH_SYMBOLS for ch in text):
            return synthesize_hybrid_line(
                text=text,
                origin_x=origin_x,
                baseline_y=baseline_y,
                model_engine=self,
                style_id=style_id,
                bias=bias,
                target_cap_height_mm=target_cap_height_mm,
                slant_deg=slant_deg,
                jitter_mm=jitter_mm,
                wobble=wobble,
                letter_spacing=letter_spacing,
                space_mult=space_mult
            )

        # 2. Leading capital 'Q' in question headers (Q1., Q2., etc.)
        prefix_strokes: List[Stroke] = []
        eff_origin_x = origin_x
        eff_text = text

        if text.startswith("Q") and len(text) > 1 and (text[1].isdigit() or text[1] in (".", ":")):
            scale_sym = target_cap_height_mm / 14.0
            q_strokes, q_adv = stroke_font.render_word_strokes(
                "Q",
                origin_x=origin_x,
                baseline_y=baseline_y,
                scale=scale_sym,
                slant_deg=slant_deg if slant_deg != 0.0 else -5.0,
                jitter_max=max(0.04, jitter_mm),
                kerning_var=0.0,
                wobble=max(0.04, wobble)
            )
            prefix_strokes = q_strokes
            eff_origin_x = origin_x + q_adv
            eff_text = text[1:].lstrip()

        # 3. Regular text: if space_mult is adjusted from 1.0, synthesize word-by-word with custom space gap
        if abs(space_mult - 1.0) > 0.02 and " " in eff_text:
            words = eff_text.split(" ")
            all_strokes = list(prefix_strokes)
            curr_x = eff_origin_x
            base_space = 2.4 * (target_cap_height_mm / 3.8) * space_mult

            for w_idx, w in enumerate(words):
                if not w:
                    curr_x += base_space
                    continue
                raw_offsets = self.get_or_sample_offsets(w, style_id=style_id, bias=bias)
                if len(raw_offsets) > 0:
                    polys, w_width = self._process_offsets_to_mm(
                        raw_offsets, origin_x=curr_x, baseline_y=baseline_y,
                        target_cap_height_mm=target_cap_height_mm, letter_spacing=letter_spacing
                    )
                    if slant_deg != 0.0 or jitter_mm > 0.0 or wobble > 0.0:
                        polys = transform_strokes(
                            polys, baseline_y=baseline_y, origin_x=curr_x,
                            slant_deg=slant_deg, scale_mult=1.0, jitter_mm=jitter_mm, wobble=wobble
                        )
                    all_strokes.extend(polys)
                    curr_x += w_width
                if w_idx < len(words) - 1:
                    curr_x += base_space

            return all_strokes, curr_x - origin_x

        # Single continuous line synthesis
        raw_offsets = self.get_or_sample_offsets(eff_text, style_id=style_id, bias=bias)
        if len(raw_offsets) == 0:
            return prefix_strokes, eff_origin_x - origin_x

        polylines, total_width = self._process_offsets_to_mm(
            raw_offsets, origin_x=eff_origin_x, baseline_y=baseline_y,
            target_cap_height_mm=target_cap_height_mm, letter_spacing=letter_spacing
        )

        if slant_deg != 0.0 or jitter_mm > 0.0 or wobble > 0.0:
            polylines = transform_strokes(
                polylines,
                baseline_y=baseline_y,
                origin_x=eff_origin_x,
                slant_deg=slant_deg,
                scale_mult=1.0,
                jitter_mm=jitter_mm,
                wobble=wobble
            )

        all_strokes = prefix_strokes + polylines
        return all_strokes, (eff_origin_x + total_width) - origin_x

    def _process_offsets_to_mm(self,
                               offsets: np.ndarray,
                               origin_x: float,
                               baseline_y: float,
                               target_cap_height_mm: float,
                               letter_spacing: float = 1.0) -> Tuple[List[Stroke], float]:
        """
        Applies drawing.py pipeline from sjvasquez/handwriting-synthesis and maps
        coordinates to physical millimeters locked to notebook ruled baselines.
        """
        offsets = offsets.copy()

        # 1. Strip priming residual flick before the first pen-up jump to the text
        eos_indices = np.where(offsets[:, 2] == 1.0)[0]
        if len(eos_indices) > 0:
            first_eos = int(eos_indices[0])
            # Priming release stroke is typically within the first 25 points
            # followed by a large pen-up transit jump (jump_dx > 2.0 or jump_mag > 3.0) to start text
            if first_eos < 25:
                if first_eos + 1 < len(offsets):
                    jump_dx = float(offsets[first_eos + 1, 0])
                    jump_mag = float(math.hypot(offsets[first_eos + 1, 0], offsets[first_eos + 1, 1]))
                    if jump_dx > 2.0 or jump_mag > 3.0:
                        offsets = offsets[first_eos + 1:]
                else:
                    offsets = offsets[first_eos + 1:]

        if len(offsets) == 0:
            return [], 0.0

        offsets[:, :2] *= 1.5

        if DRAWING_AVAILABLE:
            coords = drawing.offsets_to_coords(offsets)
            coords = drawing.denoise(coords)
            span_x = float(coords[:, 0].max() - coords[:, 0].min())
            if span_x < 3.0 or len(coords) < 15:
                coords[:, 1] -= float(np.median(coords[:, 1]))
            else:
                X = coords[:, 0].reshape(-1, 1)
                Y = coords[:, 1].reshape(-1, 1)
                X_mat = np.concatenate([np.ones([X.shape[0], 1], dtype=np.float32), X], axis=1)
                try:
                    offset, slope = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y).squeeze()
                    if abs(slope) > 0.35:
                        slope = 0.0
                        offset = float(np.median(coords[:, 1]))
                    theta = float(np.arctan(slope))
                    rot = np.array(
                        [[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]],
                        dtype=np.float32
                    )
                    coords[:, :2] = np.dot(coords[:, :2], rot) - offset
                except Exception:
                    coords[:, 1] -= float(np.median(coords[:, 1]))
        else:
            coords = np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)

        # Align coordinate orientation: flip Y so positive Y is downward
        coords[:, 1] *= -1.0

        min_x = coords[:, 0].min()
        max_x = coords[:, 0].max()

        # In Graves drawing.align coordinates (with positive Y pointing down):
        # Centerline regression is at y = 0.
        # Lowercase writing baseline is at y ~ +5.0 units.
        # Distance from baseline to cap top is ~18.0 units across all IAM styles.
        RAW_CAP_HEIGHT = 18.0
        RAW_BASELINE_Y = 5.0

        scale_y = target_cap_height_mm / RAW_CAP_HEIGHT
        # Proportional aspect ratio for natural student cursive handwriting with letter spacing adjustment
        scale_x = scale_y * 0.58 * letter_spacing

        # Convert to polylines separated by end-of-stroke (eos == 1.0)
        polylines: List[Stroke] = []
        current_stroke: Stroke = []

        for x, y, eos in coords:
            # Map to physical page millimeters
            px = origin_x + (x - min_x) * scale_x
            py = baseline_y + (y - RAW_BASELINE_Y) * scale_y
            current_stroke.append((float(px), float(py)))

            if eos == 1.0:
                if len(current_stroke) >= 2:
                    # Filter out degenerate single-point or micro-ticks (< 0.2 mm span)
                    sx = [pt[0] for pt in current_stroke]
                    sy = [pt[1] for pt in current_stroke]
                    span = math.hypot(max(sx) - min(sx), max(sy) - min(sy))
                    if span > 0.20 or len(current_stroke) >= 4:
                        polylines.append(current_stroke)
                current_stroke = []

        if current_stroke and len(current_stroke) >= 2:
            sx = [pt[0] for pt in current_stroke]
            sy = [pt[1] for pt in current_stroke]
            span = math.hypot(max(sx) - min(sx), max(sy) - min(sy))
            if span > 0.20 or len(current_stroke) >= 4:
                polylines.append(current_stroke)

        total_width = (max_x - min_x) * scale_x
        return polylines, total_width


# Global instance
VASQUEZ_ENGINE = VasquezHandwritingEngine()
