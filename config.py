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

# Baseline Organic Behavior
BASELINE_JITTER: float = 0.25     # High-frequency vertical jitter (±0.25 mm)
BASELINE_DRIFT_AMP: float = 0.20  # Low-frequency arm drift amplitude (mm)
BASELINE_DRIFT_PERIOD: float = 65.0  # Period of arm drift across page (mm)

# Slant & Italic Angle (Degrees relative to vertical, tilted forward)
SLANT_DEGREES: float = 9.0        # Natural handwriting slant
SLANT_VARIANCE: float = 2.0       # Per-word slant standard deviation
KERNING_VARIANCE: float = 0.15    # Organic character spacing variance (±0.15 mm)

# Micro-tremor / Non-robotic Pen Wobble
MICRO_WOBBLE: float = 0.04        # Subtle perpendicular displacement for straight lines (mm)
PEN_HOOK_CHANCE: float = 0.15     # Chance of subtle entry/exit flourish flick

# Style Presets
STYLE_PRESETS = {
    "neat": {
        "jitter": 0.08,
        "slant": 5.0,
        "slant_var": 1.0,
        "kerning_var": 0.06,
        "wobble": 0.02,
    },
    "natural": {
        "jitter": 0.22,
        "slant": 9.5,
        "slant_var": 2.2,
        "kerning_var": 0.14,
        "wobble": 0.04,
    },
    "casual": {
        "jitter": 0.28,
        "slant": 13.0,
        "slant_var": 3.0,
        "kerning_var": 0.20,
        "wobble": 0.06,
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
