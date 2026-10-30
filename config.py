"""
config.py - Machine, Page, and Pipeline Configuration for CopyCat
Automated text-to-handwriting G-code generator for 2D GRBL / CoreXY pen plotters.
"""

import os
from dataclasses import dataclass
from typing import Tuple

# ==============================================================================
# Page & Notebook Ruling Dimensions (Units: Millimeters)
# ==============================================================================

# Standard ISO A4 Dimensions: 210.0 mm x 297.0 mm
PAGE_WIDTH: float = 210.0
PAGE_HEIGHT: float = 297.0

# Standard Ruled Notebook Baseline Spacing (8.0 mm ruled sheet)
BASELINE_SPACING: float = 8.0

# Page Margins
MARGIN_LEFT: float = 20.0        # Left margin rule (standard notebook pink/red line)
MARGIN_RIGHT: float = 190.0      # Right cutoff margin (190 mm max printable X)
MARGIN_TOP: float = 30.0         # Top margin (first text baseline sits at ~30-38 mm)
MARGIN_BOTTOM: float = 275.0     # Bottom margin cutoff (no text beyond 275 mm)

# Answer / Block Indentation
INDENT_WIDTH: float = 8.0        # Indent distance for 'Ans:' or 4-space blocks (X = 28 mm)

# Physical Usable Area:
# Width: 190 - 20 = 170 mm
# Height: 275 - 30 = 245 mm (~30 ruled lines per page)

# ==============================================================================
# Machine & GRBL Motion Parameters
# ==============================================================================

# Pen Heights (Z axis in mm)
Z_UP: float = 3.0                # Pen travel height (pen lifted, safe travel)
Z_DOWN: float = 0.0              # Pen draw height (pen touching paper firmly)

# Feed Rates (mm / min)
TRAVEL_FEED_RATE: float = 2500.0 # Rapid XY travel feed rate when pen is UP
PLUNGE_FEED_RATE: float = 800.0  # Pen plunge/lift speed along Z axis
DRAW_FEED_RATE: float = 1500.0   # Drawing feed rate when pen is DOWN on paper

# Corner Jig & Coordinate Origin Alignment
# "TOP_LEFT": (0, 0) is at top-left corner jig; +X goes right, +Y goes down page
# "BOTTOM_LEFT": Standard Cartesian CNC origin at bottom-left corner jig
ORIGIN_CORNER: str = "TOP_LEFT"

# ==============================================================================
# Serial Communication & Hardware Settings
# ==============================================================================

# Default Serial Port (auto-adapts for Linux / macOS / Windows)
DEFAULT_PORT: str = "COM3" if os.name == "nt" else "/dev/ttyUSB0"
BAUD_RATE: int = 115200
SERIAL_TIMEOUT: float = 1.0       # Seconds for serial read timeout
GRBL_BUFFER_SIZE: int = 127       # Standard GRBL RX character ring buffer limit

# ==============================================================================
# Handwriting Organic Variations & Human Biases
# ==============================================================================

# ==============================================================================
# Handwriting Organic Variations & Human Biases
# ==============================================================================

# Baseline Organic Behavior
BASELINE_JITTER: float = 0.22     # High-frequency vertical jitter (±0.22 mm)
BASELINE_DRIFT_AMP: float = 0.20  # Low-frequency arm drift amplitude (mm)
BASELINE_DRIFT_PERIOD: float = 65.0  # Period of arm drift across page (mm)

# Slant & Italic Angle (Degrees relative to vertical, tilted forward)
SLANT_DEGREES: float = 9.5        # Natural handwriting slant
SLANT_VARIANCE: float = 2.2       # Per-word slant standard deviation
KERNING_VARIANCE: float = 0.14    # Organic character spacing variance (±0.14 mm)

# Micro-tremor / Non-robotic Pen Wobble
MICRO_WOBBLE: float = 0.04        # Subtle perpendicular displacement for straight lines (mm)
PEN_HOOK_CHANCE: float = 0.15     # Chance of subtle entry/exit flourish flick

# Comprehensive Library of Handwriting Presets
STYLE_PRESETS = {
    "natural": {
        "name": "College Student (Default)",
        "description": "Everyday balanced student handwriting with organic slant and subtle baseline drift.",
        "font_variant": "simplex",
        "jitter": 0.22,
        "slant": 9.5,
        "slant_var": 2.2,
        "kerning_var": 0.14,
        "wobble": 0.04,
        "scale_mult": 1.0,
    },
    "neat": {
        "name": "Overachiever / Neat Notebook",
        "description": "Exceptionally tidy, upright, disciplined character height, and minimal jitter.",
        "font_variant": "simplex",
        "jitter": 0.06,
        "slant": 4.0,
        "slant_var": 1.0,
        "kerning_var": 0.05,
        "wobble": 0.015,
        "scale_mult": 1.0,
    },
    "architect": {
        "name": "Draftsman / Architectural",
        "description": "Crisp technical drafting lettering, zero slant, clean geometry, and uniform kerning.",
        "font_variant": "architect",
        "jitter": 0.04,
        "slant": 0.0,
        "slant_var": 0.5,
        "kerning_var": 0.04,
        "wobble": 0.01,
        "scale_mult": 0.96,
    },
    "rushed": {
        "name": "5-Minutes-Left Exam (Messy)",
        "description": "Hurried rushed exam handwriting, steep slant, loose baselines, and noticeable tremor.",
        "font_variant": "simplex",
        "jitter": 0.38,
        "slant": 16.0,
        "slant_var": 3.5,
        "kerning_var": 0.24,
        "wobble": 0.065,
        "scale_mult": 1.02,
    },
    "casual": {
        "name": "Coffee Shop Notes",
        "description": "Relaxed and friendly, slightly rounded ascenders, and gentle baseline waviness.",
        "font_variant": "simplex",
        "jitter": 0.26,
        "slant": 12.0,
        "slant_var": 2.5,
        "kerning_var": 0.18,
        "wobble": 0.05,
        "scale_mult": 1.0,
    },
    "cursive": {
        "name": "Flowing Cursive / Script",
        "description": "Flowing script style with looped ascenders, curved descenders, and elegant slant.",
        "font_variant": "script",
        "jitter": 0.18,
        "slant": 14.5,
        "slant_var": 1.8,
        "kerning_var": 0.12,
        "wobble": 0.035,
        "scale_mult": 1.0,
    },
    "compact": {
        "name": "Condensed / Cheat Sheet",
        "description": "Small, tight lettering designed to pack maximum content onto notebook lines.",
        "font_variant": "simplex",
        "jitter": 0.12,
        "slant": 6.0,
        "slant_var": 1.2,
        "kerning_var": 0.06,
        "wobble": 0.025,
        "scale_mult": 0.82,
    },
    "childlike": {
        "name": "Elementary School Round",
        "description": "Large, bubbly, rounded letterforms with wide kerning and near-zero slant.",
        "font_variant": "simplex",
        "jitter": 0.30,
        "slant": 1.5,
        "slant_var": 1.5,
        "kerning_var": 0.25,
        "wobble": 0.055,
        "scale_mult": 1.12,
    },
    "italic": {
        "name": "Formal Calligraphic Italic",
        "description": "Pronounced, sharp forward slant with disciplined baseline and tight kerning.",
        "font_variant": "simplex",
        "jitter": 0.10,
        "slant": 18.5,
        "slant_var": 1.2,
        "kerning_var": 0.08,
        "wobble": 0.025,
        "scale_mult": 1.0,
    },
    "jittery": {
        "name": "Tremor / Caffeine Overload",
        "description": "Noticeable hand tremor and micro-wobble along every pen stroke.",
        "font_variant": "simplex",
        "jitter": 0.35,
        "slant": 10.0,
        "slant_var": 3.0,
        "kerning_var": 0.22,
        "wobble": 0.09,
        "scale_mult": 1.0,
    },
    "sloppy": {
        "name": "Doctor's Prescription",
        "description": "Fast and erratic, fluctuating baseline drift, and irregular character spacing.",
        "font_variant": "script",
        "jitter": 0.42,
        "slant": 15.0,
        "slant_var": 4.5,
        "kerning_var": 0.28,
        "wobble": 0.07,
        "scale_mult": 0.95,
    },
    # --------------------------------------------------------------------------
    # Styles from sjvasquez/handwriting-synthesis (IAM Handwriting Database)
    # --------------------------------------------------------------------------
    "rnn-0": {
        "name": "IAM Style 0 (Vengeance)",
        "description": "From sjvasquez/handwriting-synthesis: Cursive/print mix with wide character spacing.",
        "font_variant": "script",
        "jitter": 0.22,
        "slant": 9.0,
        "slant_var": 2.0,
        "kerning_var": 0.16,
        "wobble": 0.04,
        "scale_mult": 1.0,
        "rnn_style_id": 0,
    },
    "rnn-1": {
        "name": "IAM Style 1 (Times)",
        "description": "From sjvasquez/handwriting-synthesis: Neat small cursive with disciplined baselines.",
        "font_variant": "simplex",
        "jitter": 0.14,
        "slant": 10.5,
        "slant_var": 1.8,
        "kerning_var": 0.10,
        "wobble": 0.03,
        "scale_mult": 0.95,
        "rnn_style_id": 1,
    },
    "rnn-2": {
        "name": "IAM Style 2 (Crowded)",
        "description": "From sjvasquez/handwriting-synthesis: Fast connected cursive with continuous word loops.",
        "font_variant": "script",
        "jitter": 0.28,
        "slant": 14.0,
        "slant_var": 2.5,
        "kerning_var": 0.18,
        "wobble": 0.05,
        "scale_mult": 1.0,
        "rnn_style_id": 2,
    },
    "rnn-3": {
        "name": "IAM Style 3 (Finish)",
        "description": "From sjvasquez/handwriting-synthesis: Sharp angular print with upright ascenders.",
        "font_variant": "architect",
        "jitter": 0.12,
        "slant": 3.0,
        "slant_var": 1.0,
        "kerning_var": 0.08,
        "wobble": 0.02,
        "scale_mult": 0.98,
        "rnn_style_id": 3,
    },
    "rnn-4": {
        "name": "IAM Style 4 (Eased Away)",
        "description": "From sjvasquez/handwriting-synthesis: Casual loose script with relaxed descenders.",
        "font_variant": "simplex",
        "jitter": 0.25,
        "slant": 11.0,
        "slant_var": 2.2,
        "kerning_var": 0.15,
        "wobble": 0.04,
        "scale_mult": 1.0,
        "rnn_style_id": 4,
    },
    "rnn-5": {
        "name": "IAM Style 5 (Aurelius)",
        "description": "From sjvasquez/handwriting-synthesis: Upright formal print with deliberate strokes.",
        "font_variant": "architect",
        "jitter": 0.08,
        "slant": 2.0,
        "slant_var": 0.8,
        "kerning_var": 0.06,
        "wobble": 0.015,
        "scale_mult": 1.0,
        "rnn_style_id": 5,
    },
    "rnn-6": {
        "name": "IAM Style 6 (Byron)",
        "description": "From sjvasquez/handwriting-synthesis: Upright compact cursive with narrow kerning.",
        "font_variant": "script",
        "jitter": 0.15,
        "slant": 5.0,
        "slant_var": 1.4,
        "kerning_var": 0.07,
        "wobble": 0.025,
        "scale_mult": 0.92,
        "rnn_style_id": 6,
    },
    "rnn-7": {
        "name": "IAM Style 7 (I Do Not Know)",
        "description": "From sjvasquez/handwriting-synthesis: Quick light cursive with delicate pen pressure.",
        "font_variant": "script",
        "jitter": 0.22,
        "slant": 12.5,
        "slant_var": 2.0,
        "kerning_var": 0.14,
        "wobble": 0.035,
        "scale_mult": 0.96,
        "rnn_style_id": 7,
    },
    "rnn-8": {
        "name": "IAM Style 8 (Vicarage)",
        "description": "From sjvasquez/handwriting-synthesis: Tall flourishing script with elongated ascenders.",
        "font_variant": "script",
        "jitter": 0.24,
        "slant": 15.0,
        "slant_var": 2.2,
        "kerning_var": 0.16,
        "wobble": 0.04,
        "scale_mult": 1.05,
        "rnn_style_id": 8,
    },
    "rnn-9": {
        "name": "IAM Style 9 (Two-Twenty)",
        "description": "From sjvasquez/handwriting-synthesis: Disciplined rhythmic print with uniform slant.",
        "font_variant": "simplex",
        "jitter": 0.16,
        "slant": 8.0,
        "slant_var": 1.5,
        "kerning_var": 0.10,
        "wobble": 0.03,
        "scale_mult": 0.98,
        "rnn_style_id": 9,
    },
    "rnn-10": {
        "name": "IAM Style 10 (Thinking)",
        "description": "From sjvasquez/handwriting-synthesis: Forward slant script with pronounced character flow.",
        "font_variant": "script",
        "jitter": 0.26,
        "slant": 16.5,
        "slant_var": 2.8,
        "kerning_var": 0.20,
        "wobble": 0.05,
        "scale_mult": 1.02,
        "rnn_style_id": 10,
    },
    "rnn-11": {
        "name": "IAM Style 11 (Royal Family)",
        "description": "From sjvasquez/handwriting-synthesis: Rounded school script with balanced proportions.",
        "font_variant": "script",
        "jitter": 0.20,
        "slant": 7.5,
        "slant_var": 1.6,
        "kerning_var": 0.13,
        "wobble": 0.035,
        "scale_mult": 1.0,
        "rnn_style_id": 11,
    },
    "rnn-12": {
        "name": "IAM Style 12 (Selfish)",
        "description": "From sjvasquez/handwriting-synthesis: Relaxed modern handwriting with organic waviness.",
        "font_variant": "simplex",
        "jitter": 0.27,
        "slant": 11.5,
        "slant_var": 2.4,
        "kerning_var": 0.17,
        "wobble": 0.045,
        "scale_mult": 1.0,
        "rnn_style_id": 12,
    },
}

# ==============================================================================
# Pygame Visualizer Display Settings
# ==============================================================================

WINDOW_WIDTH: int = 900           # Preview window width (pixels)
WINDOW_HEIGHT: int = 1000         # Preview window height (pixels)
FPS: int = 60                     # Pygame refresh rate

# Colors (RGB Tuple)
COLOR_DESK: Tuple[int, int, int] = (24, 26, 32)            # Dark slate desk background
COLOR_PAPER: Tuple[int, int, int] = (252, 250, 246)        # Soft off-white notebook paper
COLOR_PAPER_SHADOW: Tuple[int, int, int] = (14, 15, 18)    # Drop shadow
COLOR_MARGIN_LINE: Tuple[int, int, int] = (225, 95, 95)    # Ruled red/pink left margin
COLOR_RULED_LINE: Tuple[int, int, int] = (195, 215, 238)   # Ruled light blue notebook lines
COLOR_HEADER_LINE: Tuple[int, int, int] = (170, 195, 225)  # Top header accent line
COLOR_CUTOFF_LINE: Tuple[int, int, int] = (235, 185, 185)  # Right/Bottom cutoff guides
COLOR_INK: Tuple[int, int, int] = (20, 36, 75)             # Classic ballpoint / dark gel pen ink
COLOR_INK_ACCENT: Tuple[int, int, int] = (15, 25, 55)
COLOR_HUD_BG: Tuple[int, int, int] = (30, 34, 44)
COLOR_HUD_TEXT: Tuple[int, int, int] = (230, 235, 245)
COLOR_HUD_ACCENT: Tuple[int, int, int] = (80, 190, 135)


@dataclass
class PlotterConfig:
    """Consolidated configuration container passed across modules."""
    page_width: float = PAGE_WIDTH
    page_height: float = PAGE_HEIGHT
    baseline_spacing: float = BASELINE_SPACING
    margin_left: float = MARGIN_LEFT
    margin_right: float = MARGIN_RIGHT
    margin_top: float = MARGIN_TOP
    margin_bottom: float = MARGIN_BOTTOM
    indent_width: float = INDENT_WIDTH
    
    z_up: float = Z_UP
    z_down: float = Z_DOWN
    travel_feed: float = TRAVEL_FEED_RATE
    plunge_feed: float = PLUNGE_FEED_RATE
    draw_feed: float = DRAW_FEED_RATE
    origin_corner: str = ORIGIN_CORNER
    
    port: str = DEFAULT_PORT
    baud_rate: int = BAUD_RATE
    
    baseline_jitter: float = BASELINE_JITTER
    baseline_drift_amp: float = BASELINE_DRIFT_AMP
    baseline_drift_period: float = BASELINE_DRIFT_PERIOD
    slant_degrees: float = SLANT_DEGREES
    slant_variance: float = SLANT_VARIANCE
    kerning_variance: float = KERNING_VARIANCE
    micro_wobble: float = MICRO_WOBBLE


# Default global configuration instance
DEFAULT_CONFIG = PlotterConfig()
