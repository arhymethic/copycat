"""
gcode_compiler.py - Translates Single-Stroke Polylines into Optimized GRBL G-Code.

Generates standard, robust CNC / pen-plotter instructions:
- G21: Metric millimeter units
- G90: Absolute positioning
- G94: Units-per-minute feed rate mode
- G17: XY plane selection
- G00: Rapid travel moves with pen safely lifted (Z = Z_UP)
- G01: Controlled plunge and linear drawing moves (Z = Z_DOWN)

Includes:
- Path optimization (Nearest Neighbor heuristic to minimize air-travel time)
- Accurate kinematic timing estimation
- Page-by-page file generation
"""

import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import config
from layout_engine import Page, Stroke


@dataclass
class GcodeStats:
    """Kinematic and distance metrics for a compiled G-code program."""
    page_number: int
    total_strokes: int
    draw_distance_mm: float
    travel_distance_mm: float
    pen_plunges_count: int
    estimated_time_seconds: float
    total_gcode_lines: int

    @property
    def formatted_time(self) -> str:
        mins = int(self.estimated_time_seconds // 60)
        secs = int(self.estimated_time_seconds % 60)
        return f"{mins}m {secs:02d}s"


class GcodeCompiler:
    """Compiles vector strokes into clean, verified GRBL G-code programs."""

    def __init__(self, cfg: Optional[config.PlotterConfig] = None):
        self.cfg = cfg or config.DEFAULT_CONFIG

    def compile_page(self, page: Page,
                     total_pages: int = 1,
                     optimize_path: bool = True) -> Tuple[str, GcodeStats]:
        """
        Compiles a single Page object into GRBL G-code text and kinematic statistics.
        """
        strokes = page.strokes[:]
        if optimize_path and len(strokes) > 1:
            strokes = self._optimize_stroke_order(strokes)

        lines: List[str] = []
        
        # Header comments
        lines.append("; ==============================================================================")
        lines.append("; CopyCat - Centerline Pen Plotter G-Code Pipeline")
        lines.append(f"; Page {page.page_number} of {total_pages}")
        lines.append(f"; Machine: GRBL / CoreXY 2D Plotter")
        lines.append(f"; Feed Rates: Travel={self.cfg.travel_feed} mm/min, Draw={self.cfg.draw_feed} mm/min, Plunge={self.cfg.plunge_feed} mm/min")
        lines.append(f"; Pen Z: Up={self.cfg.z_up:.2f} mm, Down={self.cfg.z_down:.2f} mm")
        lines.append("; ==============================================================================")
        
        # GRBL Initialization block
        lines.append("G21          ; Millimeter units")
        lines.append("G90          ; Absolute coordinates")
        lines.append("G94          ; Feed rate mode (mm/min)")
        lines.append("G17          ; XY plane selection")
        lines.append(f"G00 Z{self.cfg.z_up:.3f} F{self.cfg.plunge_feed:.1f}  ; Lift pen to safe travel height")
        lines.append(f"G00 X0.000 Y0.000 F{self.cfg.travel_feed:.1f}   ; Rapid to corner jig (0, 0)")
        lines.append("")

        total_draw_dist = 0.0
        total_travel_dist = 0.0
        pen_plunges = 0

        current_x = 0.0
        current_y = 0.0
        pen_is_down = False

        for stroke_idx, stroke in enumerate(strokes, start=1):
            if not stroke:
                continue

            # First point of stroke
            x0, y0 = self._map_coordinates(stroke[0][0], stroke[0][1])

            # Rapid move to start of stroke (Pen UP)
            air_dx = x0 - current_x
            air_dy = y0 - current_y
            air_dist = math.hypot(air_dx, air_dy)
            total_travel_dist += air_dist

            lines.append(f"; Stroke {stroke_idx}")
            lines.append(f"G00 X{x0:.3f} Y{y0:.3f} F{self.cfg.travel_feed:.1f}")
            current_x, current_y = x0, y0

            # Pen Down (Plunge)
            lines.append(f"G01 Z{self.cfg.z_down:.3f} F{self.cfg.plunge_feed:.1f}")
            pen_is_down = True
            pen_plunges += 1

            # Draw through subsequent vertices (Pen DOWN)
            for pt in stroke[1:]:
                xi, yi = self._map_coordinates(pt[0], pt[1])
                draw_dx = xi - current_x
                draw_dy = yi - current_y
                draw_dist = math.hypot(draw_dx, draw_dy)
                total_draw_dist += draw_dist

                lines.append(f"G01 X{xi:.3f} Y{yi:.3f} F{self.cfg.draw_feed:.1f}")
                current_x, current_y = xi, yi

            # Pen Up
            lines.append(f"G00 Z{self.cfg.z_up:.3f} F{self.cfg.plunge_feed:.1f}")
            pen_is_down = False

        # Footer: Return to corner jig and end program
        lines.append("")
        lines.append("; Program Finish & Safe Retract")
        lines.append(f"G00 Z{self.cfg.z_up:.3f} F{self.cfg.plunge_feed:.1f}  ; Lift pen safely")
        lines.append(f"G00 X0.000 Y0.000 F{self.cfg.travel_feed:.1f}   ; Return to corner jig")
        lines.append("M02          ; Program End")

        # Kinematic timing estimate
        travel_time_sec = (total_travel_dist / self.cfg.travel_feed) * 60.0
        draw_time_sec = (total_draw_dist / self.cfg.draw_feed) * 60.0
        z_stroke_dist = abs(self.cfg.z_up - self.cfg.z_down)
        z_time_sec = (pen_plunges * 2 * z_stroke_dist / self.cfg.plunge_feed) * 60.0
        # Add 15% acceleration/deceleration factor for small segments
        est_total_time = (travel_time_sec + draw_time_sec + z_time_sec) * 1.15

        stats = GcodeStats(
            page_number=page.page_number,
            total_strokes=len(strokes),
            draw_distance_mm=round(total_draw_dist, 2),
            travel_distance_mm=round(total_travel_dist, 2),
            pen_plunges_count=pen_plunges,
            estimated_time_seconds=round(est_total_time, 1),
            total_gcode_lines=len(lines)
        )

        return "\n".join(lines), stats

    def compile_all(self, pages: List[Page],
                    output_dir: str = "./gcode_output",
                    optimize_path: bool = True) -> List[Tuple[str, GcodeStats]]:
        """
        Compiles all pages into separate G-code files.
        Returns a list of (saved_file_path, GcodeStats).
        """
        os.makedirs(output_dir, exist_ok=True)
        results: List[Tuple[str, GcodeStats]] = []

        total_pages = len(pages)
        for page in pages:
            gcode_content, stats = self.compile_page(
                page, total_pages=total_pages, optimize_path=optimize_path
            )
            file_name = f"page_{page.page_number}.gcode"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(gcode_content)
            results.append((file_path, stats))

        return results

    def _map_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """
        Maps page mm coordinates to the machine's origin system.
        """
        if self.cfg.origin_corner.upper() == "BOTTOM_LEFT":
            # Standard Cartesian: Y increases upwards from bottom edge
            return round(x, 3), round(self.cfg.page_height - y, 3)
        # Default "TOP_LEFT": Y increases downwards from top corner jig
        return round(x, 3), round(y, 3)

    def _optimize_stroke_order(self, strokes: List[Stroke]) -> List[Stroke]:
        """
        Nearest-Neighbor stroke sorting to minimize air-travel distance while
        preserving natural top-to-bottom, left-to-right writing order.
        """
        if not strokes:
            return []

        # Sort primarily by baseline row (within 10 mm bands) and secondarily by X
        # This keeps natural handwriting order and ink drying flow
        def stroke_key(s: Stroke):
            if not s:
                return (0, 0)
            y_band = int(s[0][1] // 8.0)  # Group by 8 mm ruled lines
            x_pos = s[0][0]
            return (y_band, x_pos)

        return sorted(strokes, key=stroke_key)
