"""
main.py - CopyCat: Automated Text-to-Handwriting G-Code Generator & Visual Inspection Pipeline.

End-to-End Workflow:
1. Load Markdown assignment (assignment.md)
2. Select from 11+ organic handwriting styles (or customize slant/jitter/tremor)
3. Launch interactive visualizer (Pygame or Tkinter studio)
4. Two-Step Console Approval Check: "Does this look good to plot? (Y/N)"
5. (If Y): Compile verified GRBL G-code & Stream to Plotter over USB Serial
6. (If N): Safely abort without sending anything to hardware.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import config
from layout_engine import LayoutEngine
from visualizer import Visualizer
from gcode_compiler import GcodeCompiler
from streamer import GrblStreamer


BANNER = r"""
  ____                      ____      _   
 / ___|___  _ __  _   _    / ___|__ _| |_ 
| |   / _ \| '_ \| | | |  | |   / _` | __|
| |__| (_) | |_) | |_| |  | |__| (_| | |_ 
 \____\___/| .__/ \__, |   \____\__,_|\__|
           |_|    |___/                   
  Automated Text-to-Handwriting Pen Plotter Pipeline
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CopyCat: Text-to-handwriting G-code generator & visual inspection for 2D pen plotters."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="assignment.md",
        help="Path to input Markdown file (default: assignment.md)"
    )
    # UI Mode
    parser.add_argument(
        "--gui", "-g",
        action="store_true",
        help="Launch the full interactive Tkinter GUI Studio with live canvas and sliders."
    )
    parser.add_argument(
        "--ui",
        choices=["auto", "pygame", "tkinter", "none"],
        default="auto",
        help="Visual inspection backend window (default: auto; tries Pygame, then Tkinter)"
    )
    # Handwriting Styles
    parser.add_argument(
        "--style", "-s",
        choices=list(config.STYLE_PRESETS.keys()),
        default="natural",
        help="Handwriting variation style preset (default: natural)"
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List all 11 available handwriting style presets with descriptions."
    )
    parser.add_argument(
        "--interactive-style", "-i",
        action="store_true",
        help="Interactively select a handwriting style from a numbered terminal menu."
    )
    # Fine-Tuning Overrides
    parser.add_argument(
        "--slant",
        type=float,
        default=None,
        help="Override forward italic slant angle in degrees (e.g. 12.0)"
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=None,
        help="Override baseline vertical jitter in mm (e.g. 0.35)"
    )
    parser.add_argument(
        "--wobble",
        type=float,
        default=None,
        help="Override micro-wobble / pen tremor displacement in mm (e.g. 0.05)"
    )
    parser.add_argument(
        "--kerning",
        type=float,
        default=None,
        help="Override character spacing variance in mm (e.g. 0.15)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override text font scaling factor (default: ~0.34)"
    )
    # Machine & Motion
    parser.add_argument(
        "--port",
        default=config.DEFAULT_PORT,
        help=f"GRBL serial port (default: {config.DEFAULT_PORT})"
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=config.BAUD_RATE,
        help=f"Serial baud rate (default: {config.BAUD_RATE})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate plotter streaming without physical hardware attached."
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "hershey", "rnn"],
        default="auto",
        help="Stroke synthesis engine (auto, hershey, or rnn; default: auto)"
    )
    parser.add_argument(
        "--output-dir",
        default="./gcode_output",
        help="Directory to save compiled .gcode files (default: ./gcode_output)"
    )
    parser.add_argument(
        "--origin",
        choices=["TOP_LEFT", "BOTTOM_LEFT"],
        default=config.ORIGIN_CORNER,
        help="Corner jig origin alignment (default: TOP_LEFT)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip visualizer window (for automated/batch pipelines)."
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Export G-code only; do not stream to the serial port."
    )
    parser.add_argument(
        "--export-preview",
        action="store_true",
        help="Export visual inspection preview images as PNG."
    )
    return parser.parse_args()


def print_styles_catalog():
    """Prints the formatted catalog of all available handwriting styles."""
    print("=" * 86)
    print("COPYCAT HANDWRITING STYLES CATALOG")
    print("=" * 86)
    print(f"{'Key':<12} {'Display Name':<30} {'Slant':<8} {'Jitter':<9} {'Variant':<10}")
    print("-" * 86)
    for key, p in config.STYLE_PRESETS.items():
        name = p.get("name", key)
        slant = f"{p.get('slant', 0.0):.1f}°"
        jitter = f"±{p.get('jitter', 0.0):.2f}mm"
        variant = p.get("font_variant", "simplex")
        print(f"{key:<12} {name:<30} {slant:<8} {jitter:<9} {variant:<10}")
        print(f"             ↳ {p.get('description', '')}\n")
    print("=" * 86)
    print("Usage: python3 main.py assignment.md --style <key>")
    print("   or: python3 main.py assignment.md --gui (for visual live sliders)")


def prompt_style_selection() -> str:
    """Prompts the user to pick a handwriting style from an interactive menu."""
    styles = list(config.STYLE_PRESETS.keys())
    print("\nSelect a handwriting style:")
    for idx, key in enumerate(styles, start=1):
        p = config.STYLE_PRESETS[key]
        print(f"  [{idx:2d}] {p['name']:<28} - {p['description']}")
    
    while True:
        choice = input(f"\nEnter choice [1-{len(styles)}] (default 1): ").strip()
        if not choice:
            return styles[0]
        if choice.isdigit() and 1 <= int(choice) <= len(styles):
            selected = styles[int(choice) - 1]
            print(f"Selected style: '{selected}' ({config.STYLE_PRESETS[selected]['name']})\n")
            return selected
        print("Invalid choice, please enter a valid number.")


def main():
    print(BANNER)
    args = parse_arguments()

    # Style catalog listing command
    if args.list_styles:
        print_styles_catalog()
        return

    # Check if Tkinter GUI Studio was requested
    if args.gui:
        print("[INFO] Launching CopyCat Interactive Tkinter Studio...")
        try:
            import ui
            ui.launch_gui(args.input)
            return
        except Exception as e:
            print(f"[GUI ERROR] Failed to launch Tkinter GUI: {e}")
            print("[INFO] Falling back to CLI mode...")

    # Step 1: Check and load input file
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file '{input_file}' does not exist.")
        print("[INFO] Creating a sample 'assignment.md' for demonstration...")
        create_sample_assignment(input_file)
        print(f"[INFO] Created '{input_file}'. Proceeding with pipeline...")

    # Select handwriting style
    selected_style = args.style
    if args.interactive_style:
        selected_style = prompt_style_selection()

    # Load custom configuration
    cfg = config.PlotterConfig(
        port=args.port,
        baud_rate=args.baud,
        origin_corner=args.origin
    )

    # Step 2: Synthesize vector strokes locked to notebook ruling
    style_info = config.STYLE_PRESETS.get(selected_style, config.STYLE_PRESETS["natural"])
    print(f"\n[1/5] Synthesizing single-stroke trajectories from '{input_file}'...")
    print(f"      Style:   {selected_style.upper()} ({style_info['name']})")
    print(f"      Details: {style_info['description']}")
    if args.slant is not None or args.jitter is not None or args.wobble is not None:
        print(f"      Custom overrides: Slant={args.slant}°, Jitter={args.jitter}mm, Wobble={args.wobble}mm")

    layout = LayoutEngine(
        cfg=cfg,
        engine_type=args.engine,
        style=selected_style,
        custom_slant=args.slant,
        custom_jitter=args.jitter,
        custom_wobble=args.wobble,
        custom_kerning=args.kerning,
        custom_scale=args.scale
    )

    try:
        pages = layout.parse_file(input_file)
    except Exception as e:
        print(f"[ERROR] Failed to parse and synthesize strokes: {e}")
        sys.exit(1)

    total_strokes = sum(p.total_stroke_count for p in pages)
    print(f"      Generated {len(pages)} page(s) containing {total_strokes} vector strokes.")

    # Step 3: Interactive Visual Inspection
    gui_decision = None
    if not args.no_preview and args.ui != "none":
        print(f"\n[2/5] Launching interactive visualizer canvas (backend: {args.ui.upper()})...")
        print("      Inspect 8.0 mm notebook ruling, margins, and centerline vector paths.")
        print("      Controls: [<- / ->] Navigate pages | [Space] Fast-forward | [Y] Approve | [N] Reject")
        vis = Visualizer(pages, cfg=cfg)
        gui_decision = vis.show(animate=True, backend=args.ui)

    if args.export_preview or args.no_preview:
        vis = Visualizer(pages, cfg=cfg)
        vis.export_preview_images()

    # Step 4: Two-Step Approval Workflow Check
    print("\n" + "=" * 70)
    print("COPYCAT - TWO-STEP PLOT INSPECTION APPROVAL")
    print("=" * 70)
    print(f"Source Markdown:      {input_file}")
    print(f"Handwriting Style:    {selected_style} ({style_info['name']})")
    print(f"Pages to Plot:        {len(pages)}")
    print(f"Total Strokes:        {total_strokes}")
    print(f"Ruled Spacing:        {cfg.baseline_spacing} mm")
    print(f"Margins (L / R / T / B): {cfg.margin_left}mm / {cfg.margin_right}mm / {cfg.margin_top}mm / {cfg.margin_bottom}mm")
    print(f"Plotter Target Port:  {args.port} ({args.baud} baud)")
    print(f"Simulation Mode:      {'ENABLED (--dry-run)' if args.dry_run else 'DISABLED (Physical Hardware)'}")
    print("=" * 70)

    # If user pressed Y or N in the GUI window, use that or prompt in terminal
    if gui_decision is True:
        print("[APPROVAL] User approved via interactive visualizer [Y].")
        approved = True
    elif gui_decision is False:
        print("[APPROVAL] User rejected via interactive visualizer [N/ESC].")
        approved = False
    else:
        while True:
            response = input("Does this look good to plot? (Y/N): ").strip().lower()
            if response in ("y", "yes"):
                approved = True
                break
            elif response in ("n", "no"):
                approved = False
                break
            else:
                print("Please enter 'Y' to proceed or 'N' to abort.")

    if not approved:
        print("\n[ABORTED] Plotting cancelled by user.")
        print("          No commands were sent to the plotter.")
        print("          Tip: Try changing the style (--style <name>) or adjust config.py.")
        sys.exit(0)

    # Step 5: Compile verified G-code programs
    print(f"\n[3/5] Compiling verified vector strokes into clean GRBL G-code...")
    compiler = GcodeCompiler(cfg=cfg)
    compiled_files = compiler.compile_all(pages, output_dir=args.output_dir)

    print(f"      G-code exported to: {args.output_dir}")
    total_est_seconds = 0.0
    for path, stats in compiled_files:
        total_est_seconds += stats.estimated_time_seconds
        print(f"      - {os.path.basename(path)}: {stats.total_gcode_lines} lines, "
              f"Draw={stats.draw_distance_mm}mm, Travel={stats.travel_distance_mm}mm, "
              f"Est Time={stats.formatted_time}")

    mins = int(total_est_seconds // 60)
    secs = int(total_est_seconds % 60)
    print(f"      Total Estimated Plotting Time: {mins}m {secs:02d}s")

    # Step 6: Stream G-code to GRBL Plotter over USB Serial
    if args.no_stream:
        print("\n[4/5] Skipping serial streaming (--no-stream specified).")
        print("[INFO] You can manually stream the files in './gcode_output/' using UGS or streamer.py.")
        print("\nDone! CopyCat finished successfully.")
        return

    print(f"\n[4/5] Connecting to plotter streamer...")
    gcode_paths = [f[0] for f in compiled_files]
    streamer = GrblStreamer(
        port=args.port,
        baud_rate=args.baud,
        dry_run=args.dry_run,
        cfg=cfg
    )

    try:
        streamer.stream_document(gcode_paths)
    except ConnectionError as e:
        print(f"\n[SERIAL ERROR] {e}")
        print("[HINT] If physical plotter is not plugged in, re-run with '--dry-run' to test streaming.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Plotting interrupted safely by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[STREAMER ERROR] An unexpected error occurred: {e}")
        sys.exit(1)

    print("\n[5/5] Done! All pages successfully plotted.")


def create_sample_assignment(filepath: str):
    """Creates a default sample Markdown assignment demonstrating homework syntax."""
    sample_content = """# Classical Mechanics: Problem Set 2

Q1. State Newton's Second Law of Motion and explain its physical significance.
Ans: Newton's second law states that the rate of change of momentum of a body is directly proportional to the applied external force, and takes place in the direction of that force. Mathematically:
    F = d(mv)/dt = m * a
This fundamental equation establishes that acceleration is directly proportional to force and inversely proportional to mass.

Q2. Derive the work-energy theorem for a particle moving in one dimension under a variable force.
Ans: Consider a particle of mass m acted upon by a net force F(x). The work done dW along a small displacement dx is:
    dW = F * dx = m * (dv/dt) * dx = m * v * dv
Integrating both sides from initial velocity v_i to final velocity v_f:
    W = integral(m * v * dv) = (1/2) * m * v_f^2 - (1/2) * m * v_i^2
Therefore, the net work done equals the net change in kinetic energy:
    W = Delta K

Q3. Differentiate between elastic and inelastic collisions.
Ans: In an elastic collision, both total linear momentum and total kinetic energy are strictly conserved before and after impact.
In an inelastic collision, total linear momentum is conserved, but part of the kinetic energy is dissipated into internal thermal energy, acoustic sound waves, or permanent plastic deformation.
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(sample_content)


if __name__ == "__main__":
    main()
