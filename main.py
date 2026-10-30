"""
main.py - CopyCat: Automated Text-to-Handwriting G-Code Generator & Visual Inspection Pipeline.

End-to-End Workflow:
1. Load Markdown assignment (assignment.md)
2. Synthesize organic single-stroke centerline vector trajectories
3. Launch interactive Pygame inspection canvas (8 mm ruled lines, margins)
4. Two-Step Console Approval Check: "Does this look good to plot? (Y/N)"
5. (If Y): Compile verified GRBL G-code & Stream to Plotter over USB Serial
6. (If N): Safely abort without sending anything to hardware.
"""

import argparse
import os
import sys
import time

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
        "--style",
        choices=["neat", "natural", "casual"],
        default="natural",
        help="Handwriting variation style preset (default: natural)"
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
        help="Skip Pygame visualizer window (for automated/batch pipelines)."
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


def main():
    print(BANNER)
    args = parse_arguments()

    # Step 1: Check and load input file
    input_file = args.input
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file '{input_file}' does not exist.")
        print("[INFO] Creating a sample 'assignment.md' for demonstration...")
        create_sample_assignment(input_file)
        print(f"[INFO] Created '{input_file}'. Proceeding with pipeline...")

    # Load custom configuration
    cfg = config.PlotterConfig(
        port=args.port,
        baud_rate=args.baud,
        origin_corner=args.origin
    )

    # Step 2: Synthesize vector strokes locked to notebook ruling
    print(f"\n[1/5] Synthesizing single-stroke trajectories from '{input_file}'...")
    print(f"      Engine: {args.engine.upper()} | Style: {args.style.upper()}")
    layout = LayoutEngine(cfg=cfg, engine_type=args.engine)

    try:
        pages = layout.parse_file(input_file)
    except Exception as e:
        print(f"[ERROR] Failed to parse and synthesize strokes: {e}")
        sys.exit(1)

    total_strokes = sum(p.total_stroke_count for p in pages)
    print(f"      Generated {len(pages)} page(s) containing {total_strokes} vector strokes.")

    # Step 3: Interactive Visual Inspection
    gui_decision = None
    if not args.no_preview:
        print("\n[2/5] Launching interactive visualizer canvas...")
        print("      Inspect 8.0 mm notebook ruling, margins, and centerline vector paths.")
        print("      Controls: [<- / ->] Navigate pages | [Space] Fast-forward | [Y] Approve | [N] Reject")
        vis = Visualizer(pages, cfg=cfg)
        gui_decision = vis.show(animate=True)

    if args.export_preview or args.no_preview:
        vis = Visualizer(pages, cfg=cfg)
        vis.export_preview_images()

    # Step 4: Two-Step Approval Workflow Check
    print("\n" + "=" * 70)
    print("COPYCAT - TWO-STEP PLOT INSPECTION APPROVAL")
    print("=" * 70)
    print(f"Source Markdown:      {input_file}")
    print(f"Pages to Plot:        {len(pages)}")
    print(f"Total Strokes:        {total_strokes}")
    print(f"Ruled Spacing:        {cfg.baseline_spacing} mm")
    print(f"Margins (L / R / T / B): {cfg.margin_left}mm / {cfg.margin_right}mm / {cfg.margin_top}mm / {cfg.margin_bottom}mm")
    print(f"Plotter Target Port:  {args.port} ({args.baud} baud)")
    print(f"Simulation Mode:      {'ENABLED (--dry-run)' if args.dry_run else 'DISABLED (Physical Hardware)'}")
    print("=" * 70)

    # If user pressed Y or N in the Pygame window, use that or prompt in terminal
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
        print("          You can adjust config.py, edit assignment.md, and retry.")
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
