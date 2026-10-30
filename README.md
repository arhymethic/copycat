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
├── config.py             # Machine, page dimensions, margins, and 24 handwriting styles
├── layout_engine.py      # Markdown parser, word-wrapper, 8 mm ruling & pagination
├── stroke_font.py        # Self-contained single-stroke vector engine & humanizer
├── rnn_engine.py         # Direct bridge to sjvasquez/handwriting-synthesis
├── visualizer.py         # Pygame & Tkinter inspection canvas with notebook ruled lines
├── gcode_compiler.py     # GRBL G-code compiler with nearest-neighbor path optimization
├── streamer.py           # USB serial streamer with multi-page corner jig pause
├── ui.py                 # Full Tkinter Studio GUI with real-time sliders and editor
├── main.py               # Main CLI orchestrator with two-step approval check
├── handwriting-synthesis/# Cloned official repo with checkpoints (42MB) and IAM styles
│   ├── checkpoints/      # Pretrained model-17900 Graves LSTM weights
│   ├── styles/           # 13 IAM real human handwriting recordings (style-0 to 12)
│   ├── drawing.py        # Denoising, coordinate conversion, and baseline alignment
│   └── rnn.py            # Neural network implementation
├── assignment.md         # Sample homework assignment in Markdown
├── requirements.txt      # Project dependencies
├── tests/                # Comprehensive unit and integration test suite
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

CopyCat integrates directly with Sean Vasquez's recurrent neural network synthesis engine:
👉 **[https://github.com/sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)**

The entire `sjvasquez/handwriting-synthesis` repository is **already cloned locally** inside `copycat/handwriting-synthesis`:
- **Pre-trained Checkpoints (`checkpoints/model-17900.*`)**: 42 MB pretrained Alex Graves LSTM model weights.
- **IAM Handwriting Style Sequences (`styles/style-{0..12}-*.npy`)**: 13 real human handwriting styles from the IAM On-Line Handwriting Database.
- **Stroke Transformation Algorithms (`drawing.py`)**: Exact coordinate conversions (`offsets_to_coords`), linear regression baseline alignment (`align`), Savitzky-Golay denoising (`denoise`), and interpolation.

### Fallback Guarantee
CopyCat features a dual-engine architecture:
1. **Self-Contained Single-Stroke Centerline Engine**: Runs out-of-the-box in pure Python and NumPy with zero heavy dependencies. Supports 11 handcrafted handwriting personas and all 13 IAM style configurations.
2. **Graves RNN Neural Engine**: Runs live generative recurrent neural network inference using `demo.py`'s `Hand` class when TensorFlow 1.x is installed.

To enable neural inference via TensorFlow:
```bash
pip install tensorflow==1.15.0  # or tensorflow-cpu
python3 main.py assignment.md --engine rnn --style rnn-2
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

## 🎨 Interactive Tkinter Studio & GUI

In addition to the Pygame inspection canvas, CopyCat includes a **full desktop Tkinter Studio (`ui.py`)**:

```bash
# Launch Tkinter GUI Studio
python3 main.py --gui
# Or directly:
python3 ui.py assignment.md
```

### Features of the GUI Studio:
- **Live 8.0 mm Ruled Sheet Canvas**: Auto-renders your handwriting on simulated notebook paper with margins.
- **Instant Style Dropdown**: Switch between all 11 handwriting styles and see the canvas update in real time.
- **Interactive Fine-Tuning Sliders**:
  - **Slant Angle Slider** ($-5^\circ\text{ to }+25^\circ$)
  - **Baseline Jitter Slider** ($0.0\text{ to }0.50\text{ mm}$)
  - **Micro-Wobble / Tremor Slider** ($0.0\text{ to }0.12\text{ mm}$)
  - **Kerning Variance Slider** ($0.0\text{ to }0.30\text{ mm}$)
  - **Font Scale Slider** ($0.20\text{ to }0.55$)
- **Live Markdown Editor**: Pop open a text window, edit notes or equations, and click *Apply* to re-render.
- **Serial Plotter Hub**: Scan available USB serial ports, toggle dry-run simulation, and stream directly with a real-time progress bar.

---

## 🖋️ Complete Handwriting Styles Catalog

CopyCat provides **24 handwriting styles** accessible via `--style <key>`, the numbered CLI picker (`-i`), or the Tkinter GUI Studio:

### 1. Organic & Academic Personas (11 Styles)
| Style Key | Display Name | Slant | Jitter | Font Variant | Personality & Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `natural` | **College Student** *(Default)* | $9.5^\circ$ | $\pm0.22\text{ mm}$ | `simplex` | Everyday balanced student handwriting, organic baseline drift, and subtle tremor. |
| `neat` | **Overachiever / Neat Notebook** | $4.0^\circ$ | $\pm0.06\text{ mm}$ | `simplex` | Exceptionally tidy, upright, disciplined character height, minimal jitter. |
| `architect` | **Draftsman / Architectural** | $0.0^\circ$ | $\pm0.04\text{ mm}$ | `architect` | Crisp technical drafting lettering, zero slant, clean geometry, uniform kerning. |
| `rushed` | **5-Minutes-Left Exam** *(Messy)* | $16.0^\circ$ | $\pm0.38\text{ mm}$ | `simplex` | Hurried rushed exam handwriting, steep slant, loose baselines, noticeable tremor. |
| `casual` | **Coffee Shop Notes** | $12.0^\circ$ | $\pm0.26\text{ mm}$ | `simplex` | Relaxed and friendly, slightly rounded ascenders, gentle baseline waviness. |
| `cursive` | **Flowing Cursive / Script** | $14.5^\circ$ | $\pm0.18\text{ mm}$ | `script` | Flowing script style with looped ascenders, curved descenders, elegant slant. |
| `compact` | **Condensed / Cheat Sheet** | $6.0^\circ$ | $\pm0.12\text{ mm}$ | `simplex` | Small, tight lettering designed to pack maximum content onto notebook lines. |
| `childlike` | **Elementary School Round** | $1.5^\circ$ | $\pm0.30\text{ mm}$ | `simplex` | Large, bubbly, rounded letterforms with wide kerning and near-zero slant. |
| `italic` | **Formal Calligraphic Italic** | $18.5^\circ$ | $\pm0.10\text{ mm}$ | `simplex` | Pronounced, sharp forward slant with disciplined baseline and tight kerning. |
| `jittery` | **Tremor / Caffeine Overload** | $10.0^\circ$ | $\pm0.35\text{ mm}$ | `simplex` | Noticeable hand tremor and micro-wobble along every pen stroke. |
| `sloppy` | **Doctor's Prescription** | $15.0^\circ$ | $\pm0.42\text{ mm}$ | `script` | Fast and erratic, fluctuating baseline drift, irregular character spacing. |

### 2. IAM Real Human Styles from `sjvasquez/handwriting-synthesis` (13 Styles)
| Style Key | IAM Style | Slant | Jitter | Font Variant | Description & Priming Sample |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rnn-0` | **IAM Style 0 (Vengeance)** | $9.0^\circ$ | $\pm0.22\text{ mm}$ | `script` | Cursive/print mix with wide character spacing. (*"thought that vengeance"*) |
| `rnn-1` | **IAM Style 1 (Times)** | $10.5^\circ$ | $\pm0.14\text{ mm}$ | `simplex` | Neat small cursive with disciplined baselines. (*"So says the Times"*) |
| `rnn-2` | **IAM Style 2 (Crowded)** | $14.0^\circ$ | $\pm0.28\text{ mm}$ | `script` | Fast connected cursive with continuous word loops. (*"House was crowded."*) |
| `rnn-3` | **IAM Style 3 (Finish)** | $3.0^\circ$ | $\pm0.12\text{ mm}$ | `architect` | Sharp angular print with upright ascenders. (*"A stronger finish"*) |
| `rnn-4` | **IAM Style 4 (Eased Away)** | $11.0^\circ$ | $\pm0.25\text{ mm}$ | `simplex` | Casual loose script with relaxed descenders. (*"now eased away"*) |
| `rnn-5` | **IAM Style 5 (Aurelius)** | $2.0^\circ$ | $\pm0.08\text{ mm}$ | `architect` | Upright formal print with deliberate strokes. (*"Aurelius Antonius about A.D"*) |
| `rnn-6` | **IAM Style 6 (Byron)** | $5.0^\circ$ | $\pm0.15\text{ mm}$ | `script` | Upright compact cursive with narrow kerning. (*"Byron, had said"*) |
| `rnn-7` | **IAM Style 7 (I Do Not Know)** | $12.5^\circ$ | $\pm0.22\text{ mm}$ | `script` | Quick light cursive with delicate pen pressure. (*"I do not know of"*) |
| `rnn-8` | **IAM Style 8 (Vicarage)** | $15.0^\circ$ | $\pm0.24\text{ mm}$ | `script` | Tall flourishing script with elongated ascenders. (*"Holmbridge Vicarage,"*) |
| `rnn-9` | **IAM Style 9 (Two-Twenty)** | $8.0^\circ$ | $\pm0.16\text{ mm}$ | `simplex` | Disciplined rhythmic print with uniform slant. (*"two-twenty carried, for him,"*) |
| `rnn-10` | **IAM Style 10 (Thinking)** | $16.5^\circ$ | $\pm0.26\text{ mm}$ | `script` | Forward slant script with pronounced character flow. (*"\"Just you keep thinking"*) |
| `rnn-11` | **IAM Style 11 (Royal Family)** | $7.5^\circ$ | $\pm0.20\text{ mm}$ | `script` | Rounded school script with balanced proportions. (*"Royal family so often seems to find"*) |
| `rnn-12` | **IAM Style 12 (Selfish)** | $11.5^\circ$ | $\pm0.27\text{ mm}$ | `simplex` | Relaxed modern handwriting with organic waviness. (*"tomorrow. How selfish of me. It"*) |

To list all styles in the terminal:
```bash
python3 main.py --list-styles
```

---

## 💻 CLI Reference

```text
usage: main.py [-h] [--gui] [--ui {auto,pygame,tkinter,none}]
               [--style {natural,neat,architect,rushed,casual,cursive,compact,childlike,italic,jittery,sloppy}]
               [--list-styles] [--interactive-style]
               [--slant SLANT] [--jitter JITTER] [--wobble WOBBLE]
               [--kerning KERNING] [--scale SCALE]
               [--port PORT] [--baud BAUD] [--dry-run]
               [--engine {auto,hershey,rnn}] [--output-dir OUTPUT_DIR]
               [--origin {TOP_LEFT,BOTTOM_LEFT}]
               [--no-preview] [--no-stream] [--export-preview]
               [input]

positional arguments:
  input                 Path to Markdown file (default: assignment.md)

options:
  -h, --help            Show this help message and exit
  --gui, -g             Launch the interactive Tkinter GUI Studio with live sliders
  --ui {auto,pygame,tkinter,none}
                        Visual inspection backend (default: auto)
  --style, -s STYLE     Handwriting style preset (default: natural)
  --list-styles         Print all 11 handwriting styles with descriptions
  --interactive-style, -i
                        Interactively pick a style from a numbered menu
  --slant SLANT         Override slant angle in degrees (e.g. 14.0)
  --jitter JITTER       Override baseline jitter in mm (e.g. 0.35)
  --wobble WOBBLE       Override micro-wobble / tremor in mm (e.g. 0.05)
  --kerning KERNING     Override character spacing variance in mm (e.g. 0.15)
  --scale SCALE         Override font scale (e.g. 0.38)
  --port PORT           GRBL serial port (default: /dev/ttyUSB0 or COM3)
  --baud BAUD           Serial baud rate (default: 115200)
  --dry-run             Simulate USB streaming without physical hardware
  --engine {auto,hershey,rnn}
                        Stroke engine: auto, hershey, or rnn (default: auto)
  --output-dir OUTPUT_DIR
                        Directory to save compiled .gcode files (default: ./gcode_output)
  --origin {TOP_LEFT,BOTTOM_LEFT}
                        Corner jig origin alignment (default: TOP_LEFT)
  --no-preview          Skip visualizer (for automated scripting)
  --no-stream           Export G-code only; do not stream to serial port
  --export-preview      Export high-resolution PNG inspection preview images
```

### Common Commands:

1. **Launch Interactive Tkinter Studio**:
   ```bash
   python3 main.py --gui
   ```
2. **List All Handwriting Styles**:
   ```bash
   python3 main.py --list-styles
   ```
3. **Pick a Style (e.g. Architect, Cursive, Rushed)**:
   ```bash
   python3 main.py assignment.md --style architect
   python3 main.py assignment.md --style cursive
   python3 main.py assignment.md --style rushed
   ```
4. **Interactive Style Menu in Terminal**:
   ```bash
   python3 main.py assignment.md -i
   ```
5. **Fine-Tuned Custom Handwriting**:
   ```bash
   python3 main.py assignment.md --style natural --slant 14.0 --jitter 0.30 --wobble 0.05
   ```
6. **Dry-Run Simulation**:
   ```bash
   python3 main.py assignment.md --dry-run
   ```
7. **Export G-Code Only**:
   ```bash
   python3 main.py assignment.md --no-stream
   ```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

