# CopyCat 🖋️🐱
### Automated Text-to-Handwriting G-Code Generator & Visual Inspection Pipeline for 2D GRBL / CoreXY Pen Plotters

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)](https://python.org)
[![GRBL: 1.1+](https://img.shields.io/badge/GRBL-1.1+-orange.svg)](https://github.com/gnea/grbl)
[![Pygame](https://img.shields.io/badge/Preview-Pygame-red.svg)](https://pygame.org)

**CopyCat** is an end-to-end Python pipeline that converts structured Markdown assignments, notes, and homework into human-like handwritten vector trajectories for 2D pen plotters (GRBL / CoreXY / AxiDraw-style machines). 

It features an **interactive Pygame visual inspection window**, **simulated 8.0 mm ruled notebook paper with margins**, **organic human handwriting variations**, a **two-step approval safety check**, and **USB serial streaming with physical corner jig multi-page handling**.

---

## ⚡ The Centerline Vector Difference

### Why Outline TTF/OTF Fonts Fail on Pen Plotters
Standard digital fonts (`.ttf`, `.otf`, system fonts) describe **filled 2D closed polygon outlines**. When sent to a physical pen plotter with a ballpoint, gel pen, or fountain pen:
- The pen traces both the outer and inner perimeter of each letter, producing **hollow, robotic "bubble letters"**.
- Doubled strokes tear notebook paper and look blatantly artificial.

### The CopyCat Solution: True Single-Stroke Centerline Paths
CopyCat synthesizes **true 1D centerline vector paths** (single strokes). The pen makes a single, continuous, natural stroke per character stroke—just like a human hand holding a pen. 

```
Standard Outline TTF:       CopyCat Centerline Vector:
      ┌─────┐                          |
      │     │                          |
      │     │                          |───┐
      │     │                          |   │
      └─────┘                          |───┘
 (Hollow bubble trace)            (Authentic pen stroke)
```

Furthermore, CopyCat applies **organic human variations**:
- **±0.25 mm high-frequency baseline jitter**
- **Low-frequency arm drift** (subtle waviness as the hand moves across the page)
- **Natural italic slant variance** per word (5° to 14°)
- **Kerning fluctuation**
- **Micro-tremor / micro-wobble** along straight segments to eliminate robotic straightness.

---

## 📋 Table of Contents
- [Architecture & Pipeline](#-architecture--pipeline)
- [Two-Step Approval Workflow](#-two-step-approval-workflow)
- [Repository Structure](#-repository-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Hardware Setup & Corner Jig](#-hardware-setup--corner-jig)
- [Markdown Syntax Guide](#-markdown-syntax-guide)
- [Neural Handwriting Synthesis Integration](#-neural-handwriting-synthesis-integration)
- [Configuration (`config.py`)](#-configuration-configpy)
- [CLI Reference](#-cli-reference)
- [License](#-license)

---

## 🏗️ Architecture & Pipeline

```
  ┌─────────────────┐
  │  assignment.md  │  (Structured Markdown: # Headers, Q1., Ans:, Indents)
  └────────┬────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────┐
  │                 layout_engine.py                       │
  │  - Single-stroke vector synthesis                      │
  │  - 8.0 mm ruled baseline alignment                     │
  │  - Margin boundaries (20 mm to 190 mm)                 │
  │  - Auto word-wrapping & A4 multi-page pagination       │
  │  - Bias profiles (neat for questions, natural answers) │
  └────────┬───────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────┐
  │                  visualizer.py                         │
  │  - Interactive Pygame window                           │
  │  - Canvas: 8 mm notebook ruling, margins, drop shadow  │
  │  - Real-time animated vector trajectory drawing        │
  │  - Page navigation (<- / -> arrow keys)                │
  └────────┬───────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────┐
  │           TWO-STEP APPROVAL SAFETY CHECK               │
  │        "Does this look good to plot? (Y/N)"            │
  └────────┬───────────────────────────────┬───────────────┘
           │ [Y]                           │ [N]
           ▼                               ▼
  ┌──────────────────┐           ┌──────────────────┐
  │ gcode_compiler   │           │ Abort & Exit     │
  │ Clean GRBL G-code│           │ (No hardware     │
  │ (G21, G90, G00,  │           │  commands sent)  │
  │  G01, Z_UP/DOWN) │           └──────────────────┘
  └────────┬─────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────┐
  │                   streamer.py                          │
  │  - Direct USB Serial communication (pyserial, 115200)  │
  │  - GRBL character-counting & 'ok' response loop        │
  │  - Real-time progress bar & ETA                        │
  │  - Corner jig pause between pages (sheet replacement)  │
  │  - Emergency stop safety (Ctrl+C lifts pen immediately)│
  └────────────────────────────────────────────────────────┘
```

---

## 🛡️ Two-Step Approval Workflow

CopyCat strictly enforces a **"verify-before-send"** safety philosophy. It **never** streams blindly to physical hardware:

1. **Step 1: Synthesis & Inspection**:
   CopyCat parses the document, synthesizes vector strokes, and launches an interactive Pygame visualization window showing the exact simulated notebook page, ruled lines, margins, and pen paths. You can page through multi-page assignments using `<-` and `->` keys.
2. **Step 2: Explicit User Confirmation**:
   The terminal prompts:
   ```text
   ======================================================================
   COPYCAT - TWO-STEP PLOT INSPECTION APPROVAL
   ======================================================================
   Source Markdown:      assignment.md
   Pages to Plot:        2
   Total Strokes:        3482
   Ruled Spacing:        8.0 mm
   Margins (L / R / T / B): 20.0mm / 190.0mm / 30.0mm / 275.0mm
   Plotter Target Port:  /dev/ttyUSB0 (115200 baud)
   Simulation Mode:      DISABLED (Physical Hardware)
   ======================================================================
   Does this look good to plot? (Y/N): 
   ```
   - Press **`Y`**: G-code is compiled and streamed to the plotter.
   - Press **`N`**: The job is safely cancelled. No G-code is sent to the machine.

---

## 📁 Repository Structure

```
copycat/
├── config.py             # Machine, page dimensions, margins, and motion parameters
├── layout_engine.py      # Markdown parser, word-wrapper, 8 mm ruling & pagination
├── stroke_font.py        # Self-contained single-stroke vector engine & humanizer
├── rnn_engine.py         # Integration adapter for sjvasquez/handwriting-synthesis
├── visualizer.py         # Pygame inspection canvas with notebook ruled lines
├── gcode_compiler.py     # GRBL G-code compiler with nearest-neighbor path optimization
├── streamer.py           # USB serial streamer with multi-page corner jig pause
├── main.py               # Main CLI orchestrator
├── assignment.md         # Sample homework assignment in Markdown
├── requirements.txt      # Project dependencies
└── README.md             # Complete documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/copycat.git
cd copycat
```

### 2. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Grant Serial Port Permissions (Linux)
If running on Linux, add your user to the `dialout` or `uucp` group to access USB serial devices without sudo:
```bash
sudo usermod -a -G dialout $USER
# Log out and log back in for changes to take effect
```

### 4. Run a Dry-Run Simulation (No Hardware Needed!)
Test the full pipeline out-of-the-box in simulation mode:
```bash
python3 main.py assignment.md --dry-run
```

---

## 📐 Hardware Setup & Corner Jig

CopyCat is designed for standard 2D pen plotters and CNC drawing machines:
- **Kinematics**: CoreXY or Cartesian (X/Y gantry) running GRBL 1.1 firmware.
- **Z-Axis**: Micro servo, solenoid, or stepper motor moving the pen between `Z_UP` (travel) and `Z_DOWN` (draw).
- **Physical Corner Jig**:
  To support seamless multi-page plotting, place a physical 90-degree corner jig (e.g. 3D-printed L-bracket or taped ruler) at the origin `(0, 0)` of your plotter bed:
  - Top-left edge of the paper is pushed flush into the jig.
  - When Page 1 finishes, CopyCat lifts the pen and parks it at `(0, 0)`.
  - CopyCat prompts you in the terminal to remove Page 1, align a fresh sheet into the jig, and press **[Enter]** to resume Page 2!

```
      Corner Jig (0, 0)
      ┌───────────══════════════════════════════════════════
      │  ┌───────────────────────────────────────────────┐
      │  │                                               │
      │  │  Top Margin (30 mm)                           │
      │  │ ───────────────────────────────────────────── │
      │  │ │                                             │
      │  │ │ 8.0 mm Ruled Line                           │
      │  │ │ 8.0 mm Ruled Line                           │
      │  │ │                                             │
      │  │ │                                             │
      │  │ │ Margin Left (20 mm)    Right Cutoff (190 mm)│
      │  │ │                                             │
      │  └─┴─────────────────────────────────────────────┘
      │    A4 Sheet (210 mm x 297 mm)
```

---

## ✍️ Markdown Syntax Guide

Create your homework or notes in standard Markdown (`assignment.md`). CopyCat automatically structures the handwriting based on formatting cues:

| Markdown Syntax | Layout Behavior | Handwriting Style |
| :--- | :--- | :--- |
| `# Title` | **Centered horizontally** on page, double line jump | Slightly larger scale, proud/neat |
| `Q1.`, `Q2.`, `1.` | **Left-aligned at 20 mm margin** | Neater bias, upright slant (5°), low jitter |
| `Ans:` or `Answer:` | **Indented by 8 mm (starts at 28 mm)** | Natural bias, relaxed slant (9°-13°), organic jitter |
| 4-space indent (`    `) | **Indented block / Equations** | Natural bias, indented continuation |
| Blank line | **Single line jump** | Advances down by one 8.0 mm notebook line |
| Long paragraphs | **Automatic word-wrapping** | Wraps at 190 mm cutoff, paginates at 275 mm |

### Sample `assignment.md`:
```markdown
# Physics Assignment: Dynamics

Q1. State Newton's Second Law of Motion.
Ans: Newton's second law states that the rate of change of momentum of an object is directly proportional to the applied force, and takes place in the direction of that force. Mathematically:
    F = m * a

Q2. What is work done by a constant force?
Ans: The work done by a constant force is defined as the scalar dot product of the force vector and the displacement vector:
    W = F . d = |F| * |d| * cos(theta)
```

---

## 🧠 Neural Handwriting Synthesis Integration

CopyCat integrates with Sean Vasquez's recurrent neural network synthesis engine:
👉 **[https://github.com/sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)**

### Fallback Guarantee
CopyCat ships with a **100% self-contained single-stroke Hershey vector engine**. If `handwriting-synthesis` weights are not installed, CopyCat runs immediately out-of-the-box without requiring gigabytes of checkpoints.

### Enabling the Graves RNN Engine:
1. Clone the `handwriting-synthesis` repository into the `copycat/` folder or install its module:
   ```bash
   git clone https://github.com/sjvasquez/handwriting-synthesis.git
   ```
2. Download pretrained model checkpoints into `checkpoints/` as described in its repository.
3. Run CopyCat with the neural engine flag:
   ```bash
   python3 main.py assignment.md --engine rnn
   ```

---

## ⚙️ Configuration (`config.py`)

All machine, page, and handwriting parameters are centralized in [`config.py`](config.py):

```python
# Page & Notebook Dimensions (Millimeters)
PAGE_WIDTH = 210.0          # ISO A4 width
PAGE_HEIGHT = 297.0         # ISO A4 height
BASELINE_SPACING = 8.0      # Standard notebook ruled line spacing
MARGIN_LEFT = 20.0          # Left margin line
MARGIN_RIGHT = 190.0        # Right cutoff margin
MARGIN_TOP = 30.0           # Top margin
MARGIN_BOTTOM = 275.0       # Bottom margin cutoff
INDENT_WIDTH = 8.0          # Indent for 'Ans:' (starts at 28.0 mm)

# Machine Parameters
Z_UP = 3.0                  # Pen lifted safe travel height (mm)
Z_DOWN = 0.0                # Pen drawing contact height (mm)
TRAVEL_FEED_RATE = 2500.0   # Rapid travel feed rate (mm/min)
PLUNGE_FEED_RATE = 800.0    # Pen plunge/lift rate (mm/min)
DRAW_FEED_RATE = 1500.0     # Drawing feed rate (mm/min)

# Hardware Serial Settings
DEFAULT_PORT = "/dev/ttyUSB0"  # (or "COM3" on Windows)
BAUD_RATE = 115200

# Handwriting Organic Variations
BASELINE_JITTER = 0.25      # High-frequency vertical jitter (±0.25 mm)
BASELINE_DRIFT_AMP = 0.20   # Arm drift amplitude across notebook (mm)
SLANT_DEGREES = 9.0         # Italic forward slant angle
KERNING_VARIANCE = 0.15     # Character spacing variance (mm)
MICRO_WOBBLE = 0.04         # Anti-robotic hand tremor (mm)
```

---

## 💻 CLI Reference

```text
usage: main.py [-h] [--port PORT] [--baud BAUD] [--dry-run]
               [--engine {auto,hershey,rnn}] [--style {neat,natural,casual}]
               [--output-dir OUTPUT_DIR] [--origin {TOP_LEFT,BOTTOM_LEFT}]
               [--no-preview] [--no-stream] [--export-preview]
               [input]

positional arguments:
  input                 Path to Markdown file (default: assignment.md)

options:
  -h, --help            Show this help message and exit
  --port PORT           GRBL serial port (default: /dev/ttyUSB0 or COM3)
  --baud BAUD           Serial baud rate (default: 115200)
  --dry-run             Simulate USB streaming without physical hardware
  --engine {auto,hershey,rnn}
                        Stroke engine: auto, hershey, or rnn (default: auto)
  --style {neat,natural,casual}
                        Handwriting style preset (default: natural)
  --output-dir OUTPUT_DIR
                        Directory to save compiled .gcode files (default: ./gcode_output)
  --origin {TOP_LEFT,BOTTOM_LEFT}
                        Corner jig origin alignment (default: TOP_LEFT)
  --no-preview          Skip Pygame visualizer (for automated scripting)
  --no-stream           Export G-code only; do not stream to serial port
  --export-preview      Export high-resolution PNG inspection preview images
```

### Common Commands:

1. **Standard Workflow (Preview + Approval + Stream)**:
   ```bash
   python3 main.py assignment.md
   ```
2. **Export G-Code Only (e.g. for Universal Gcode Sender or SD card)**:
   ```bash
   python3 main.py assignment.md --no-stream
   ```
3. **Dry-Run Simulation**:
   ```bash
   python3 main.py assignment.md --dry-run
   ```
4. **Neat Handwriting Style**:
   ```bash
   python3 main.py assignment.md --style neat
   ```
5. **Custom Port on Windows**:
   ```bash
   python main.py assignment.md --port COM4
   ```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
