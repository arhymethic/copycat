"""
ui.py - Interactive Tkinter GUI Studio for CopyCat.

Features:
1. Live Notebook Canvas: Renders realistic 8.0 mm ruled notebook paper, margins,
   and single-stroke handwriting trajectories in real-time.
2. Rich Handwriting Style Library: Instant dropdown selection across 11 distinct
   handwriting styles (Student, Overachiever, Draftsman, Rushed Exam, Cursive Script, etc.).
3. Fine-Tuning Sliders: Real-time interactive adjustment of Slant, Baseline Jitter,
   Micro-Wobble / Tremor, Kerning, and Scale.
4. Embedded Markdown Editor: Edit or write notes live with instant preview update.
5. G-Code Exporter & USB Streamer: One-click export and USB serial plotter streaming.
"""

import math
import os
import sys
import threading
import time
from typing import List, Optional

import config
import layout_engine
import gcode_compiler
import streamer

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class CopyCatStudio:
    """Tkinter Desktop Studio for CopyCat Pen Plotter Pipeline."""

    def __init__(self, root: tk.Tk, initial_file: str = "assignment.md"):
        self.root = root
        self.root.title("CopyCat - Handwriting Pen Plotter Studio")
        self.root.geometry("1200x850")
        self.root.minsize(950, 700)

        # State
        self.cfg = config.PlotterConfig()
        self.current_filepath = initial_file if os.path.exists(initial_file) else "assignment.md"
        self.markdown_content = ""
        self.pages: List[layout_engine.Page] = []
        self.current_page_idx = 0
        self.zoom_level = 1.0

        # Handwriting parameter variables
        self.style_var = tk.StringVar(value="natural")
        self.slant_var = tk.DoubleVar(value=config.STYLE_PRESETS["natural"]["slant"])
        self.jitter_var = tk.DoubleVar(value=config.STYLE_PRESETS["natural"]["jitter"])
        self.wobble_var = tk.DoubleVar(value=config.STYLE_PRESETS["natural"]["wobble"])
        self.kerning_var = tk.DoubleVar(value=config.STYLE_PRESETS["natural"]["kerning_var"])
        self.scale_var = tk.DoubleVar(value=0.34)

        # Serial settings
        self.port_var = tk.StringVar(value=self.cfg.port)
        self.baud_var = tk.IntVar(value=self.cfg.baud_rate)
        self.dry_run_var = tk.BooleanVar(value=True)

        # Style description variable
        self.style_desc_var = tk.StringVar(value=config.STYLE_PRESETS["natural"]["description"])

        # Setup UI
        self._load_initial_file()
        self._build_ui()
        self.update_pipeline()

    def _load_initial_file(self):
        """Loads file content or generates sample content if absent."""
        if os.path.exists(self.current_filepath):
            with open(self.current_filepath, "r", encoding="utf-8") as f:
                self.markdown_content = f.read()
        else:
            self.markdown_content = (
                "# Classical Mechanics: Problem Set 2\n\n"
                "Q1. State Newton's Second Law of Motion.\n"
                "Ans: Newton's second law states that the rate of change of momentum of a body "
                "is directly proportional to the applied external force: F = m * a.\n\n"
                "Q2. Derive the work-energy theorem.\n"
                "Ans: The work done dW along displacement dx is:\n"
                "    dW = F * dx = m * v * dv\n"
                "Integrating gives: W = Delta K.\n"
            )

    def _build_ui(self):
        """Constructs the responsive left sidebar and right canvas view."""
        # Top-level paned layout
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ----------------------------------------------------------------------
        # Left Sidebar (Controls)
        # ----------------------------------------------------------------------
        sidebar_frame = ttk.Frame(paned, padding=12, width=380)
        paned.add(sidebar_frame, weight=0)

        # Document Section
        doc_group = ttk.LabelFrame(sidebar_frame, text="📄 Document Source", padding=8)
        doc_group.pack(fill=tk.X, pady=(0, 8))

        file_row = ttk.Frame(doc_group)
        file_row.pack(fill=tk.X)
        self.file_label = ttk.Label(file_row, text=os.path.basename(self.current_filepath), font=("sans-serif", 10, "bold"))
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_browse = ttk.Button(file_row, text="Browse...", command=self._on_browse_file)
        btn_browse.pack(side=tk.RIGHT, padx=(4, 0))

        btn_edit = ttk.Button(doc_group, text="✏️ Edit Markdown Live", command=self._on_open_editor)
        btn_edit.pack(fill=tk.X, pady=(6, 0))

        # Handwriting Style Section (The user's core request!)
        style_group = ttk.LabelFrame(sidebar_frame, text="🖋️ Handwriting Style", padding=8)
        style_group.pack(fill=tk.X, pady=(0, 8))

        # Style dropdown
        style_choices = list(config.STYLE_PRESETS.keys())
        style_cb = ttk.Combobox(
            style_group,
            textvariable=self.style_var,
            values=style_choices,
            state="readonly",
            font=("sans-serif", 10, "bold")
        )
        style_cb.pack(fill=tk.X)
        style_cb.bind("<<ComboboxSelected>>", self._on_style_selected)

        desc_lbl = ttk.Label(
            style_group,
            textvariable=self.style_desc_var,
            wraplength=340,
            font=("sans-serif", 9),
            foreground="#444455"
        )
        desc_lbl.pack(fill=tk.X, pady=(6, 4))

        # Fine-Tuning Parameter Sliders
        sliders_group = ttk.LabelFrame(sidebar_frame, text="🎛️ Organic Sliders", padding=8)
        sliders_group.pack(fill=tk.X, pady=(0, 8))

        # Slant slider
        self._create_slider(sliders_group, "Slant Angle (°):", self.slant_var, -5.0, 25.0, 0.5)
        # Jitter slider
        self._create_slider(sliders_group, "Baseline Jitter (mm):", self.jitter_var, 0.0, 0.50, 0.02)
        # Wobble slider
        self._create_slider(sliders_group, "Micro-Wobble / Tremor:", self.wobble_var, 0.0, 0.12, 0.005)
        # Kerning slider
        self._create_slider(sliders_group, "Kerning Variance (mm):", self.kerning_var, 0.0, 0.30, 0.01)
        # Font Scale
        self._create_slider(sliders_group, "Text Scale:", self.scale_var, 0.20, 0.55, 0.01)

        btn_reset = ttk.Button(sliders_group, text="Reset to Style Defaults", command=self._on_reset_sliders)
        btn_reset.pack(fill=tk.X, pady=(6, 2))

        # Hardware & Serial Section
        plotter_group = ttk.LabelFrame(sidebar_frame, text="🔌 Machine & Serial Port", padding=8)
        plotter_group.pack(fill=tk.X, pady=(0, 8))

        port_row = ttk.Frame(plotter_group)
        port_row.pack(fill=tk.X)
        ttk.Label(port_row, text="Port:").pack(side=tk.LEFT)
        self.port_entry = ttk.Entry(port_row, textvariable=self.port_var, width=14)
        self.port_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(port_row, text="Scan", width=6, command=self._on_scan_ports).pack(side=tk.LEFT)

        baud_row = ttk.Frame(plotter_group)
        baud_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(baud_row, text="Baud:").pack(side=tk.LEFT)
        baud_cb = ttk.Combobox(baud_row, textvariable=self.baud_var, values=[115200, 250000, 57600], width=10, state="readonly")
        baud_cb.pack(side=tk.LEFT, padx=4)

        dry_run_chk = ttk.Checkbutton(plotter_group, text="Dry-Run Simulation (No Hardware)", variable=self.dry_run_var)
        dry_run_chk.pack(anchor=tk.W, pady=(4, 0))

        # Action Buttons
        btn_render = ttk.Button(sidebar_frame, text="🔄 Re-render Canvas", command=self.update_pipeline)
        btn_render.pack(fill=tk.X, pady=(0, 4))

        btn_export = ttk.Button(sidebar_frame, text="💾 Export G-Code (.gcode)", command=self._on_export_gcode)
        btn_export.pack(fill=tk.X, pady=(0, 4))

        btn_stream = ttk.Button(sidebar_frame, text="🚀 Stream to Pen Plotter", command=self._on_stream_plotter)
        btn_stream.pack(fill=tk.X, pady=(0, 4))

        # ----------------------------------------------------------------------
        # Right Main Canvas (Live Ruled Sheet Preview)
        # ----------------------------------------------------------------------
        canvas_container = ttk.Frame(paned)
        paned.add(canvas_container, weight=1)

        # Toolbar above canvas
        toolbar = ttk.Frame(canvas_container, padding=6)
        toolbar.pack(fill=tk.X)

        self.btn_prev = ttk.Button(toolbar, text="◀ Prev Page", command=self._on_prev_page)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.page_indicator = ttk.Label(toolbar, text="Page 1 of 1", font=("sans-serif", 10, "bold"))
        self.page_indicator.pack(side=tk.LEFT, padx=10)

        self.btn_next = ttk.Button(toolbar, text="Next Page ▶", command=self._on_next_page)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        # Zoom controls
        ttk.Button(toolbar, text="Zoom In (+)", command=lambda: self._adjust_zoom(1.15)).pack(side=tk.LEFT, padx=(20, 2))
        ttk.Button(toolbar, text="Zoom Out (-)", command=lambda: self._adjust_zoom(0.85)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit Width", command=self._fit_width).pack(side=tk.LEFT, padx=2)

        self.stats_label = ttk.Label(toolbar, text="Strokes: 0", font=("sans-serif", 9), foreground="#555566")
        self.stats_label.pack(side=tk.RIGHT, padx=10)

        # Drawing Canvas
        self.canvas = tk.Canvas(canvas_container, bg="#181a20", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_page())

    def _create_slider(self, parent: ttk.Frame, label_text: str, variable: tk.DoubleVar,
                       min_val: float, max_val: float, step: float):
        """Creates a labeled slider linked to instant re-render."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        
        val_lbl = ttk.Label(row, text=f"{variable.get():.2f}", width=5)
        lbl = ttk.Label(row, text=label_text, width=18)
        lbl.pack(side=tk.LEFT)
        val_lbl.pack(side=tk.RIGHT)

        def on_change(val):
            val_lbl.config(text=f"{float(val):.2f}")
            self.root.after_idle(self.update_pipeline)

        scale = ttk.Scale(
            parent,
            from_=min_val,
            to=max_val,
            variable=variable,
            orient=tk.HORIZONTAL,
            command=on_change
        )
        scale.pack(fill=tk.X, pady=(0, 4))

    def _on_style_selected(self, event=None):
        """Called when a new style preset is picked from the dropdown."""
        key = self.style_var.get()
        if key in config.STYLE_PRESETS:
            preset = config.STYLE_PRESETS[key]
            self.style_desc_var.set(preset["description"])
            self.slant_var.set(preset["slant"])
            self.jitter_var.set(preset["jitter"])
            self.wobble_var.set(preset["wobble"])
            self.kerning_var.set(preset["kerning_var"])
            self.scale_var.set(0.34 * preset.get("scale_mult", 1.0))
            self.update_pipeline()

    def _on_reset_sliders(self):
        """Resets sliders to current style default values."""
        self._on_style_selected()

    def _adjust_zoom(self, factor: float):
        self.zoom_level = max(0.4, min(3.0, self.zoom_level * factor))
        self._draw_page()

    def _fit_width(self):
        canv_w = self.canvas.winfo_width()
        if canv_w > 100:
            self.zoom_level = (canv_w - 60) / (self.cfg.page_width * 2.5)
            self._draw_page()

    def _on_prev_page(self):
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self._draw_page()

    def _on_next_page(self):
        if self.current_page_idx < len(self.pages) - 1:
            self.current_page_idx += 1
            self._draw_page()

    def _on_scan_ports(self):
        ports = streamer.GrblStreamer.list_available_ports()
        if ports:
            self.port_var.set(ports[0])
            messagebox.showinfo("Ports Detected", f"Found serial ports: {', '.join(ports)}")
        else:
            messagebox.showinfo("No Hardware Found", "No USB serial plotters detected. Check cables or use Dry-Run mode.")

    def _on_browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Markdown Document",
            filetypes=[("Markdown Files", "*.md"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if path:
            self.current_filepath = path
            self.file_label.config(text=os.path.basename(path))
            with open(path, "r", encoding="utf-8") as f:
                self.markdown_content = f.read()
            self.update_pipeline()

    def _on_open_editor(self):
        """Opens a live markdown editor popup window."""
        win = tk.Toplevel(self.root)
        win.title("Live Markdown Editor - CopyCat")
        win.geometry("700x550")

        txt = tk.Text(win, wrap=tk.WORD, font=("monospace", 11), padx=8, pady=8)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", self.markdown_content)

        btn_row = ttk.Frame(win, padding=6)
        btn_row.pack(fill=tk.X)

        def apply_changes():
            self.markdown_content = txt.get("1.0", tk.END)
            self.update_pipeline()
            win.destroy()

        ttk.Button(btn_row, text="Apply & Re-render Canvas", command=apply_changes).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)

    def update_pipeline(self):
        """Synthesizes vector trajectories from Markdown and re-renders canvas."""
        layout = layout_engine.LayoutEngine(
            cfg=self.cfg,
            style=self.style_var.get(),
            custom_slant=self.slant_var.get(),
            custom_jitter=self.jitter_var.get(),
            custom_wobble=self.wobble_var.get(),
            custom_kerning=self.kerning_var.get(),
            custom_scale=self.scale_var.get()
        )

        try:
            self.pages = layout.parse_markdown(self.markdown_content)
        except Exception as e:
            messagebox.showerror("Parsing Error", f"Failed to parse document: {e}")
            return

        if self.current_page_idx >= len(self.pages):
            self.current_page_idx = max(0, len(self.pages) - 1)

        self._draw_page()

    def _draw_page(self):
        """Draws the realistic notebook paper, margins, and vector strokes on Tkinter canvas."""
        self.canvas.delete("all")
        if not self.pages:
            return

        current_page = self.pages[self.current_page_idx]
        total_pages = len(self.pages)

        # Update indicators
        self.page_indicator.config(text=f"Page {self.current_page_idx + 1} of {total_pages}")
        self.btn_prev.config(state=tk.NORMAL if self.current_page_idx > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_page_idx < total_pages - 1 else tk.DISABLED)
        self.stats_label.config(text=f"Strokes: {current_page.total_stroke_count}  |  Lines: {current_page.line_count}")

        canv_w = self.canvas.winfo_width()
        canv_h = self.canvas.winfo_height()
        if canv_w < 50 or canv_h < 50:
            return

        # Base scale: maps mm to pixels
        base_px_per_mm = 2.4 * self.zoom_level
        paper_w = self.cfg.page_width * base_px_per_mm
        paper_h = self.cfg.page_height * base_px_per_mm

        # Center paper on canvas
        paper_x = max(20.0, (canv_w - paper_w) / 2.0)
        paper_y = 25.0

        # 1. Paper Drop Shadow
        self.canvas.create_rectangle(
            paper_x + 6, paper_y + 8, paper_x + paper_w + 6, paper_y + paper_h + 8,
            fill="#101115", outline=""
        )

        # 2. Paper Sheet (Ivory notebook)
        self.canvas.create_rectangle(
            paper_x, paper_y, paper_x + paper_w, paper_y + paper_h,
            fill="#FCFAF6", outline="#D0CECA"
        )

        # 3. Horizontal 8.0 mm Ruled Lines (Light blue)
        y_mm = self.cfg.margin_top
        while y_mm <= self.cfg.margin_bottom:
            py = paper_y + y_mm * base_px_per_mm
            self.canvas.create_line(
                paper_x, py, paper_x + paper_w, py,
                fill="#C3D7EE", width=1
            )
            y_mm += self.cfg.baseline_spacing

        # 4. Top Margin Rule (Header accent)
        top_py = paper_y + self.cfg.margin_top * base_px_per_mm
        self.canvas.create_line(paper_x, top_py, paper_x + paper_w, top_py, fill="#A8C5E8", width=1)

        # 5. Left Vertical Margin Line (Notebook Pink/Red)
        margin_px = paper_x + self.cfg.margin_left * base_px_per_mm
        self.canvas.create_line(
            margin_px, paper_y, margin_px, paper_y + paper_h,
            fill="#E15F5F", width=2
        )

        # 6. Right & Bottom Cutoff Boundary Guides (Subtle dashed guides)
        cutoff_right_px = paper_x + self.cfg.margin_right * base_px_per_mm
        cutoff_bottom_py = paper_y + self.cfg.margin_bottom * base_px_per_mm
        self.canvas.create_line(cutoff_right_px, top_py, cutoff_right_px, cutoff_bottom_py, fill="#E6C8C8", dash=(3, 3))
        self.canvas.create_line(margin_px, cutoff_bottom_py, cutoff_right_px, cutoff_bottom_py, fill="#E6C8C8", dash=(3, 3))

        # 7. Render Vector Trajectories in Dark Ink Blue
        ink_color = "#14244B"
        for stroke in current_page.strokes:
            if len(stroke) < 2:
                continue
            coords: List[float] = []
            for px, py in stroke:
                coords.append(paper_x + px * base_px_per_mm)
                coords.append(paper_y + py * base_px_per_mm)
            self.canvas.create_line(coords, fill=ink_color, width=2, capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def _on_export_gcode(self):
        """Compiles and exports G-code files."""
        out_dir = filedialog.askdirectory(title="Select Output Directory for G-Code")
        if not out_dir:
            return
        compiler = gcode_compiler.GcodeCompiler(cfg=self.cfg)
        results = compiler.compile_all(self.pages, output_dir=out_dir)
        total_time = sum(r[1].estimated_time_seconds for r in results)
        mins = int(total_time // 60)
        secs = int(total_time % 60)
        messagebox.showinfo(
            "Export Complete",
            f"Successfully compiled {len(results)} G-code file(s) into:\n{out_dir}\n"
            f"Total estimated plot time: {mins}m {secs:02d}s"
        )

    def _on_stream_plotter(self):
        """Streams compiled G-code to the GRBL pen plotter."""
        compiler = gcode_compiler.GcodeCompiler(cfg=self.cfg)
        compiled_files = compiler.compile_all(self.pages, output_dir="./gcode_output")
        gcode_paths = [f[0] for f in compiled_files]

        confirm = messagebox.askyesno(
            "Confirm Plotting",
            f"Ready to plot {len(gcode_paths)} page(s) to {self.port_var.get()}?\n"
            f"Dry Run: {'YES' if self.dry_run_var.get() else 'NO (Physical Hardware)'}\n\n"
            "Proceed?"
        )
        if not confirm:
            return

        # Launch progress window
        prog_win = tk.Toplevel(self.root)
        prog_win.title("Streaming to Pen Plotter")
        prog_win.geometry("500x200")

        lbl_status = ttk.Label(prog_win, text="Connecting to GRBL...", font=("sans-serif", 10, "bold"))
        lbl_status.pack(pady=(20, 8))

        prog_bar = ttk.Progressbar(prog_win, mode="determinate", length=400)
        prog_bar.pack(pady=10)

        def run_streaming():
            st = streamer.GrblStreamer(
                port=self.port_var.get(),
                baud_rate=self.baud_var.get(),
                dry_run=self.dry_run_var.get(),
                cfg=self.cfg
            )
            try:
                st.connect()
                for idx, path in enumerate(gcode_paths, start=1):
                    lbl_status.config(text=f"Plotting Page {idx} of {len(gcode_paths)}...")
                    prog_bar["value"] = (idx - 1) / len(gcode_paths) * 100
                    st.stream_file(path, page_number=idx, total_pages=len(gcode_paths))
                prog_bar["value"] = 100
                lbl_status.config(text="Done! All pages successfully plotted.")
                messagebox.showinfo("Plot Complete", "All pages have been successfully plotted!")
                prog_win.destroy()
            except Exception as e:
                messagebox.showerror("Streaming Error", f"Plotting failed: {e}")
                prog_win.destroy()

        threading.Thread(target=run_streaming, daemon=True).start()


def launch_gui(filepath: str = "assignment.md"):
    """Entry point to launch the Tkinter Studio."""
    if not TKINTER_AVAILABLE:
        print("[ERROR] Tkinter is not available in this Python environment.")
        sys.exit(1)
    root = tk.Tk()
    app = CopyCatStudio(root, initial_file=filepath)
    root.mainloop()


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else "assignment.md"
    launch_gui(initial)

