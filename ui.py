"""
ui.py - Professional Dark Mode Studio for CopyCat Pen Plotter Pipeline.

Features:
1. Live Interactive Canvas: Realistic notebook ruling, margins, and vector trajectories.
   Full pan (click-and-drag), mouse wheel zoom, coordinate inspection HUD.
2. Real-Time Plot Simulation: Step-by-step pen drawing animation playback.
3. Dual-Panel Interface:
   - Embedded Markdown Editor with disk sync and auto-reload on file change.
   - Style library and granular handwriting parameter sliders.
4. Preserved Parameter Tuning: Switching handwriting styles keeps custom slider settings.
5. Professional Dark Mode: High-contrast, clean industrial CAD/CAM appearance without emojis.
"""

import math
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

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


# Theme color palette (Professional Deep Charcoal / Midnight Dark)
DARK_BG = "#14151f"
DARK_PANEL = "#1b1c28"
DARK_PANEL_ALT = "#222434"
DARK_BORDER = "#2c2e42"
DARK_TEXT = "#e2e5f0"
DARK_MUTED = "#868ba3"
DARK_INPUT = "#101119"
DARK_ACCENT = "#4f86f7"
DARK_ACCENT_HOVER = "#6b9bf9"
DARK_BTN = "#252738"
DARK_BTN_HOVER = "#32354c"
PAPER_BG = "#fbfaf6"
RULING_BLUE = "#d8e5f2"
MARGIN_RED = "#e57373"
INK_NAVY = "#1a1f3c"


class CopyCatStudio:
    """Professional Pen Plotter Desktop Studio for CopyCat."""

    def __init__(self, root: tk.Tk, initial_file: str = "assignment.md"):
        self.root = root
        self.root.title("CopyCat - Pen Plotter Studio")
        self.root.geometry("1300x880")
        self.root.minsize(1000, 720)
        self.root.configure(bg=DARK_BG)

        # Pipeline state
        self.cfg = config.PlotterConfig()
        self.current_filepath = os.path.abspath(initial_file if os.path.exists(initial_file) else "assignment.md")
        self.last_mtime = 0.0
        self.markdown_content = ""
        self.pages: List[layout_engine.Page] = []
        self.current_page_idx = 0

        # Interactive Canvas state (Pan & Zoom)
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._is_dragging = False

        # Simulation animation state
        self.sim_animating = False
        self.sim_stroke_idx = 0
        self.sim_speed = 5  # strokes per frame

        # Style & parameter variables
        self.style_var = tk.StringVar(value="natural")
        self.slant_var = tk.DoubleVar(value=config.STYLE_PRESETS["natural"]["slant"])
        self.jitter_var = tk.DoubleVar(value=config.STYLE_PRESETS["natural"]["jitter"])
        self.wobble_var = tk.DoubleVar(value=config.STYLE_PRESETS["natural"]["wobble"])
        self.kerning_var = tk.DoubleVar(value=config.STYLE_PRESETS["natural"]["kerning_var"])
        self.scale_var = tk.DoubleVar(value=0.34)
        self.style_desc_var = tk.StringVar(value=config.STYLE_PRESETS["natural"]["description"])

        # Serial settings
        self.port_var = tk.StringVar(value=self.cfg.port)
        self.baud_var = tk.IntVar(value=self.cfg.baud_rate)
        self.dry_run_var = tk.BooleanVar(value=True)

        # Coordinate HUD
        self.hud_coords_var = tk.StringVar(value="X: 0.0 mm | Y: 0.0 mm")

        # Build application
        self._setup_styles()
        self._load_file_from_disk()
        self._build_ui()
        self.update_pipeline()

        # Start background disk watcher
        self._schedule_file_watch()

    def _setup_styles(self):
        """Applies clean dark theme across all ttk widgets."""
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.style.configure(".", background=DARK_BG, foreground=DARK_TEXT, borderwidth=0)
        self.style.configure("TFrame", background=DARK_BG)
        self.style.configure("Panel.TFrame", background=DARK_PANEL)
        self.style.configure("PanelAlt.TFrame", background=DARK_PANEL_ALT)

        self.style.configure("TLabel", background=DARK_BG, foreground=DARK_TEXT, font=("sans-serif", 9))
        self.style.configure("Panel.TLabel", background=DARK_PANEL, foreground=DARK_TEXT, font=("sans-serif", 9))
        self.style.configure("Muted.TLabel", background=DARK_PANEL, foreground=DARK_MUTED, font=("sans-serif", 8))
        self.style.configure("Header.TLabel", background=DARK_PANEL, foreground=DARK_TEXT, font=("sans-serif", 10, "bold"))
        self.style.configure("Accent.TLabel", background=DARK_PANEL, foreground=DARK_ACCENT, font=("sans-serif", 9, "bold"))

        self.style.configure("TButton",
                             background=DARK_BTN,
                             foreground=DARK_TEXT,
                             font=("sans-serif", 9),
                             padding=(8, 4),
                             borderwidth=1,
                             relief="flat")
        self.style.map("TButton",
                       background=[("active", DARK_BTN_HOVER), ("disabled", DARK_BG)],
                       foreground=[("disabled", DARK_MUTED)])

        self.style.configure("Primary.TButton",
                             background=DARK_ACCENT,
                             foreground="#ffffff",
                             font=("sans-serif", 9, "bold"),
                             padding=(8, 5))
        self.style.map("Primary.TButton",
                       background=[("active", DARK_ACCENT_HOVER)])

        self.style.configure("TNotebook", background=DARK_PANEL, borderwidth=0)
        self.style.configure("TNotebook.Tab",
                             background=DARK_PANEL_ALT,
                             foreground=DARK_MUTED,
                             padding=(12, 6),
                             font=("sans-serif", 9))
        self.style.map("TNotebook.Tab",
                       background=[("selected", DARK_PANEL)],
                       foreground=[("selected", DARK_TEXT)])

        self.style.configure("TCombobox",
                             fieldbackground=DARK_INPUT,
                             background=DARK_BTN,
                             foreground=DARK_TEXT,
                             selectbackground=DARK_ACCENT,
                             selectforeground="#ffffff",
                             font=("sans-serif", 9))

        self.style.configure("TCheckbutton", background=DARK_PANEL, foreground=DARK_TEXT)
        self.style.map("TCheckbutton", background=[("active", DARK_PANEL)])

        self.style.configure("TScale", background=DARK_PANEL, troughcolor=DARK_INPUT)

    def _load_file_from_disk(self) -> bool:
        """Loads file content from disk and tracks modification timestamp."""
        if os.path.exists(self.current_filepath):
            try:
                mtime = os.path.getmtime(self.current_filepath)
                with open(self.current_filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                self.markdown_content = content
                self.last_mtime = mtime
                return True
            except Exception as e:
                print(f"[ERROR] Failed to read {self.current_filepath}: {e}")
                return False
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
            return False

    def _schedule_file_watch(self):
        """Polls disk every 1000ms. If external editor saves changes, auto-reloads."""
        if hasattr(self, "root") and self.root.winfo_exists():
            try:
                if os.path.exists(self.current_filepath):
                    mtime = os.path.getmtime(self.current_filepath)
                    if mtime > self.last_mtime:
                        self._load_file_from_disk()
                        if hasattr(self, "editor_txt"):
                            self.editor_txt.delete("1.0", tk.END)
                            self.editor_txt.insert("1.0", self.markdown_content)
                        self.update_pipeline()
            except Exception:
                pass
            self.root.after(1000, self._schedule_file_watch)

    def _build_ui(self):
        """Constructs split-pane interface with dark theme layout."""
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=DARK_BORDER, bd=1, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True)

        # ----------------------------------------------------------------------
        # Left Panel (Sidebar with Notebook Tabs)
        # ----------------------------------------------------------------------
        left_container = ttk.Frame(paned, style="Panel.TFrame", width=420)
        paned.add(left_container, minsize=360)

        # Header bar
        header_bar = ttk.Frame(left_container, style="Panel.TFrame", padding=(12, 10))
        header_bar.pack(fill=tk.X)

        title_lbl = ttk.Label(header_bar, text="COPYCAT", font=("sans-serif", 13, "bold"), foreground=DARK_ACCENT)
        title_lbl.pack(side=tk.LEFT)
        sub_lbl = ttk.Label(header_bar, text="  Plotter Studio", style="Muted.TLabel")
        sub_lbl.pack(side=tk.LEFT)

        # Notebook tabs
        self.notebook = ttk.Notebook(left_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Tab 1: Editor & Source
        tab_editor = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
        self.notebook.add(tab_editor, text="Markdown Source")

        # Tab 2: Style & Tuning
        tab_style = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
        self.notebook.add(tab_style, text="Handwriting Style")

        # Tab 3: Hardware & Streamer
        tab_hardware = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
        self.notebook.add(tab_hardware, text="Machine & Serial")

        self._build_editor_tab(tab_editor)
        self._build_style_tab(tab_style)
        self._build_hardware_tab(tab_hardware)

        # ----------------------------------------------------------------------
        # Right Panel (Interactive Canvas & Viewport)
        # ----------------------------------------------------------------------
        right_container = ttk.Frame(paned, style="TFrame")
        paned.add(right_container, minsize=500)

        self._build_canvas_view(right_container)

    def _build_editor_tab(self, parent: ttk.Frame):
        """Embedded Markdown source editor with disk reload & re-render buttons."""
        file_row = ttk.Frame(parent, style="Panel.TFrame")
        file_row.pack(fill=tk.X, pady=(0, 8))

        self.file_label = ttk.Label(file_row, text=os.path.basename(self.current_filepath), style="Header.TLabel")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_browse = ttk.Button(file_row, text="Browse", command=self._on_browse_file)
        btn_browse.pack(side=tk.RIGHT, padx=(4, 0))

        btn_reload = ttk.Button(file_row, text="Reload Disk", command=self._on_reload_disk)
        btn_reload.pack(side=tk.RIGHT)

        # Editor text area
        text_frame = tk.Frame(parent, bg=DARK_BORDER, bd=1)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.editor_txt = tk.Text(
            text_frame,
            wrap=tk.WORD,
            bg=DARK_INPUT,
            fg=DARK_TEXT,
            insertbackground=DARK_ACCENT,
            selectbackground=DARK_ACCENT,
            selectforeground="#ffffff",
            font=("monospace", 10),
            padx=8,
            pady=8,
            bd=0,
            undo=True
        )
        self.editor_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.editor_txt.insert("1.0", self.markdown_content)

        scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.editor_txt.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.editor_txt.configure(yscrollcommand=scroll.set)

        # Action bar
        action_bar = ttk.Frame(parent, style="Panel.TFrame")
        action_bar.pack(fill=tk.X)

        btn_apply = ttk.Button(action_bar, text="Apply Changes & Re-render", style="Primary.TButton", command=self._on_apply_editor)
        btn_apply.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_save = ttk.Button(action_bar, text="Save to Disk", command=self._on_save_editor_to_disk)
        btn_save.pack(side=tk.RIGHT)

    def _build_style_tab(self, parent: ttk.Frame):
        """Handwriting style dropdown and persistent fine-tuning sliders."""
        # Style selector
        lbl_style = ttk.Label(parent, text="Handwriting Style Preset", style="Header.TLabel")
        lbl_style.pack(anchor=tk.W, pady=(0, 4))

        style_choices = list(config.STYLE_PRESETS.keys())
        self.style_cb = ttk.Combobox(
            parent,
            textvariable=self.style_var,
            values=style_choices,
            state="readonly",
            font=("sans-serif", 10, "bold")
        )
        self.style_cb.pack(fill=tk.X, pady=(0, 4))
        self.style_cb.bind("<<ComboboxSelected>>", self._on_style_selected)

        desc_lbl = ttk.Label(
            parent,
            textvariable=self.style_desc_var,
            wraplength=380,
            style="Muted.TLabel"
        )
        desc_lbl.pack(fill=tk.X, pady=(0, 12))

        # Fine-tuning sliders
        sliders_lbl = ttk.Label(parent, text="Fine-Tuning Parameters", style="Header.TLabel")
        sliders_lbl.pack(anchor=tk.W, pady=(0, 4))

        slider_box = ttk.Frame(parent, style="Panel.TFrame")
        slider_box.pack(fill=tk.X, pady=(0, 8))

        self._create_slider(slider_box, "Slant Angle (°):", self.slant_var, -5.0, 32.0)
        self._create_slider(slider_box, "Baseline Jitter (mm):", self.jitter_var, 0.0, 0.50)
        self._create_slider(slider_box, "Micro-Wobble / Tremor:", self.wobble_var, 0.0, 0.12)
        self._create_slider(slider_box, "Kerning Variance (mm):", self.kerning_var, 0.0, 0.30)
        self._create_slider(slider_box, "Text Scale:", self.scale_var, 0.20, 0.55)

        btn_reset = ttk.Button(parent, text="Reset Sliders to Style Defaults", command=self._on_reset_sliders)
        btn_reset.pack(fill=tk.X, pady=(4, 12))

        # Re-render button
        btn_render = ttk.Button(parent, text="Re-render Canvas", style="Primary.TButton", command=self.update_pipeline)
        btn_render.pack(fill=tk.X)

    def _build_hardware_tab(self, parent: ttk.Frame):
        """Machine connection settings and G-code export/streaming controls."""
        sec_lbl = ttk.Label(parent, text="Machine Communication", style="Header.TLabel")
        sec_lbl.pack(anchor=tk.W, pady=(0, 8))

        port_row = ttk.Frame(parent, style="Panel.TFrame")
        port_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(port_row, text="Serial Port:", style="Panel.TLabel", width=12).pack(side=tk.LEFT)
        port_entry = ttk.Entry(port_row, textvariable=self.port_var, width=16)
        port_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(port_row, text="Scan Ports", command=self._on_scan_ports).pack(side=tk.LEFT)

        baud_row = ttk.Frame(parent, style="Panel.TFrame")
        baud_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(baud_row, text="Baud Rate:", style="Panel.TLabel", width=12).pack(side=tk.LEFT)
        baud_cb = ttk.Combobox(baud_row, textvariable=self.baud_var, values=[115200, 250000, 57600], width=12, state="readonly")
        baud_cb.pack(side=tk.LEFT, padx=4)

        dry_chk = ttk.Checkbutton(parent, text="Dry-Run Simulation Mode (No hardware)", variable=self.dry_run_var)
        dry_chk.pack(anchor=tk.W, pady=(6, 14))

        btn_export = ttk.Button(parent, text="Export GRBL G-Code", command=self._on_export_gcode)
        btn_export.pack(fill=tk.X, pady=(0, 6))

        btn_stream = ttk.Button(parent, text="Stream to Plotter Hardware", style="Primary.TButton", command=self._on_stream_plotter)
        btn_stream.pack(fill=tk.X)

    def _build_canvas_view(self, parent: ttk.Frame):
        """Constructs interactive viewport toolbar, canvas, and status HUD."""
        # Viewport Toolbar
        toolbar = ttk.Frame(parent, style="Panel.TFrame", padding=(10, 6))
        toolbar.pack(fill=tk.X)

        self.btn_prev = ttk.Button(toolbar, text="< Previous", command=self._on_prev_page)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.page_indicator = ttk.Label(toolbar, text="Page 1 of 1", style="Header.TLabel")
        self.page_indicator.pack(side=tk.LEFT, padx=10)

        self.btn_next = ttk.Button(toolbar, text="Next >", command=self._on_next_page)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        # Zoom & Navigation buttons
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="Zoom In (+)", command=lambda: self._adjust_zoom(1.20)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Zoom Out (-)", command=lambda: self._adjust_zoom(0.80)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit Width", command=self._fit_width).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit Page", command=self._fit_page).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="100% Reset", command=self._reset_view).pack(side=tk.LEFT, padx=2)

        # Simulation animation button
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.btn_sim = ttk.Button(toolbar, text="Simulate Plotting", command=self._toggle_simulation)
        self.btn_sim.pack(side=tk.LEFT, padx=2)

        self.stats_label = ttk.Label(toolbar, text="Strokes: 0", style="Muted.TLabel")
        self.stats_label.pack(side=tk.RIGHT, padx=6)

        # Drawing Canvas
        self.canvas = tk.Canvas(parent, bg="#0d0e14", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas Event Bindings
        self.canvas.bind("<Configure>", lambda e: self._draw_page())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<MouseWheel>", self._on_canvas_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._adjust_zoom(1.15))
        self.canvas.bind("<Button-5>", lambda e: self._adjust_zoom(0.85))

        # Bottom Status HUD
        status_hud = ttk.Frame(parent, style="Panel.TFrame", padding=(10, 4))
        status_hud.pack(fill=tk.X)

        self.hud_label = ttk.Label(status_hud, textvariable=self.hud_coords_var, style="Muted.TLabel")
        self.hud_label.pack(side=tk.LEFT)

        guide_lbl = ttk.Label(status_hud, text="Drag to Pan | Scroll to Zoom | Click Simulate to preview pen travel", style="Muted.TLabel")
        guide_lbl.pack(side=tk.RIGHT)

    def _create_slider(self, parent: ttk.Frame, label_text: str, variable: tk.DoubleVar,
                       min_val: float, max_val: float):
        """Creates slider linked to instant re-rendering without resetting other parameters."""
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=2)

        lbl = ttk.Label(row, text=label_text, style="Panel.TLabel", width=20)
        lbl.pack(side=tk.LEFT)

        val_lbl = ttk.Label(row, text=f"{variable.get():.2f}", style="Accent.TLabel", width=6)
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
        """
        When user changes handwriting style, PRESERVE current slider settings
        and apply them immediately to the new style.
        """
        key = self.style_var.get()
        if key in config.STYLE_PRESETS:
            preset = config.STYLE_PRESETS[key]
            self.style_desc_var.set(preset["description"])
            # DO NOT overwrite user sliders! Apply current parameters to the new style!
            self.update_pipeline()

    def _on_reset_sliders(self):
        """Explicitly reset sliders to the active style defaults."""
        key = self.style_var.get()
        if key in config.STYLE_PRESETS:
            preset = config.STYLE_PRESETS[key]
            self.slant_var.set(preset["slant"])
            self.jitter_var.set(preset["jitter"])
            self.wobble_var.set(preset["wobble"])
            self.kerning_var.set(preset["kerning_var"])
            self.scale_var.set(0.34 * preset.get("scale_mult", 1.0))
            self.update_pipeline()

    def _on_apply_editor(self):
        """Applies text from the embedded editor, updates state, and re-renders."""
        self.markdown_content = self.editor_txt.get("1.0", tk.END).strip()
        self.update_pipeline()

    def _on_save_editor_to_disk(self):
        """Saves current editor content to disk."""
        self.markdown_content = self.editor_txt.get("1.0", tk.END).strip()
        try:
            with open(self.current_filepath, "w", encoding="utf-8") as f:
                f.write(self.markdown_content)
            self.last_mtime = os.path.getmtime(self.current_filepath)
            messagebox.showinfo("Saved", f"Successfully saved to {os.path.basename(self.current_filepath)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not write file: {e}")

    def _on_reload_disk(self):
        """Explicitly reloads assignment content from disk."""
        if self._load_file_from_disk():
            self.editor_txt.delete("1.0", tk.END)
            self.editor_txt.insert("1.0", self.markdown_content)
            self.update_pipeline()
            messagebox.showinfo("Reloaded", f"Loaded latest contents of {os.path.basename(self.current_filepath)}")
        else:
            messagebox.showerror("Error", f"File not found on disk: {self.current_filepath}")

    def _on_browse_file(self):
        """Opens file browser to select a different assignment file."""
        path = filedialog.askopenfilename(
            title="Select Markdown Document",
            filetypes=[("Markdown Files", "*.md"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if path:
            self.current_filepath = os.path.abspath(path)
            self.file_label.config(text=os.path.basename(path))
            self._load_file_from_disk()
            self.editor_txt.delete("1.0", tk.END)
            self.editor_txt.insert("1.0", self.markdown_content)
            self.update_pipeline()

    def update_pipeline(self):
        """Synthesizes single-stroke vector trajectories and updates page view."""
        # Sync editor content if available
        if hasattr(self, "editor_txt"):
            txt_val = self.editor_txt.get("1.0", tk.END).strip()
            if txt_val:
                self.markdown_content = txt_val

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
            messagebox.showerror("Parsing Error", f"Failed to layout document: {e}")
            return

        if self.current_page_idx >= len(self.pages):
            self.current_page_idx = max(0, len(self.pages) - 1)

        # Reset animation index when document changes
        self.sim_stroke_idx = 0
        self._draw_page()

    def _draw_page(self):
        """Draws realistic notebook paper, margins, and single-stroke trajectories."""
        self.canvas.delete("all")
        if not self.pages:
            return

        current_page = self.pages[self.current_page_idx]
        total_pages = len(self.pages)

        # Update indicators
        self.page_indicator.config(text=f"Page {self.current_page_idx + 1} of {total_pages}")
        self.btn_prev.config(state=tk.NORMAL if self.current_page_idx > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_page_idx < total_pages - 1 else tk.DISABLED)
        self.stats_label.config(text=f"Strokes: {current_page.total_stroke_count} | Lines: {current_page.line_count}")

        canv_w = self.canvas.winfo_width()
        canv_h = self.canvas.winfo_height()
        if canv_w < 50 or canv_h < 50:
            return

        base_px_per_mm = 2.4 * self.zoom_level
        paper_w = self.cfg.page_width * base_px_per_mm
        paper_h = self.cfg.page_height * base_px_per_mm

        # Center paper on canvas + user pan offsets
        center_x = (canv_w - paper_w) / 2.0 + self.pan_x
        center_y = 25.0 + self.pan_y

        self._paper_origin_x = center_x
        self._paper_origin_y = center_y
        self._px_per_mm = base_px_per_mm

        # 1. Drop shadow
        self.canvas.create_rectangle(
            center_x + 8, center_y + 10, center_x + paper_w + 8, center_y + paper_h + 10,
            fill="#08090d", outline=""
        )

        # 2. Paper Sheet (Ivory notebook sheet)
        self.canvas.create_rectangle(
            center_x, center_y, center_x + paper_w, center_y + paper_h,
            fill=PAPER_BG, outline="#2b2d3d", width=1
        )

        # 3. Horizontal 8.0 mm Ruled Notebook Lines (Light blue)
        y_mm = self.cfg.margin_top
        while y_mm <= self.cfg.margin_bottom + 5.0:
            py = center_y + y_mm * base_px_per_mm
            self.canvas.create_line(
                center_x, py, center_x + paper_w, py,
                fill=RULING_BLUE, width=1
            )
            y_mm += self.cfg.baseline_spacing

        # 4. Vertical Margin Line (Notebook red line at 20 mm)
        margin_x = center_x + self.cfg.margin_left * base_px_per_mm
        self.canvas.create_line(
            margin_x, center_y, margin_x, center_y + paper_h,
            fill=MARGIN_RED, width=1
        )

        # 5. Right Margin Boundary Guideline (Dotted line at 190 mm)
        cutoff_x = center_x + self.cfg.margin_right * base_px_per_mm
        self.canvas.create_line(
            cutoff_x, center_y + self.cfg.margin_top * base_px_per_mm,
            cutoff_x, center_y + self.cfg.margin_bottom * base_px_per_mm,
            fill="#e2e8f0", dash=(2, 4)
        )

        # 6. Render Single-Stroke Vector Trajectories
        strokes_to_draw = current_page.strokes
        if self.sim_animating:
            strokes_to_draw = current_page.strokes[:self.sim_stroke_idx]

        for stroke in strokes_to_draw:
            if len(stroke) < 2:
                continue
            coords = []
            for x_mm, y_mm in stroke:
                coords.append(center_x + x_mm * base_px_per_mm)
                coords.append(center_y + y_mm * base_px_per_mm)

            self.canvas.create_line(
                *coords,
                fill=INK_NAVY,
                width=max(1, int(1.2 * self.zoom_level)),
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND
            )

        # If simulating, draw virtual pen tip cursor
        if self.sim_animating and strokes_to_draw and len(strokes_to_draw[-1]) >= 1:
            last_pt = strokes_to_draw[-1][-1]
            tip_x = center_x + last_pt[0] * base_px_per_mm
            tip_y = center_y + last_pt[1] * base_px_per_mm
            self.canvas.create_oval(
                tip_x - 3, tip_y - 3, tip_x + 3, tip_y + 3,
                fill="#ff4757", outline="#ffffff", width=1
            )

    # --------------------------------------------------------------------------
    # Interactive Viewport Navigation (Pan & Zoom)
    # --------------------------------------------------------------------------
    def _on_canvas_press(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._is_dragging = True

    def _on_canvas_drag(self, event):
        if self._is_dragging:
            dx = event.x - self._drag_start_x
            dy = event.y - self._drag_start_y
            self.pan_x += dx
            self.pan_y += dy
            self._drag_start_x = event.x
            self._drag_start_y = event.y
            self._draw_page()

    def _on_canvas_release(self, event):
        self._is_dragging = False

    def _on_canvas_wheel(self, event):
        if event.delta > 0:
            self._adjust_zoom(1.15)
        else:
            self._adjust_zoom(0.85)

    def _on_canvas_motion(self, event):
        """Displays real-time millimeter coordinates on hover."""
        if hasattr(self, "_paper_origin_x") and hasattr(self, "_px_per_mm") and self._px_per_mm > 0:
            x_mm = (event.x - self._paper_origin_x) / self._px_per_mm
            y_mm = (event.y - self._paper_origin_y) / self._px_per_mm
            if 0.0 <= x_mm <= self.cfg.page_width and 0.0 <= y_mm <= self.cfg.page_height:
                line_idx = max(0, int((y_mm - self.cfg.margin_top) / self.cfg.baseline_spacing) + 1)
                self.hud_coords_var.set(f"X: {x_mm:5.1f} mm  |  Y: {y_mm:5.1f} mm  |  Ruled Line: {line_idx}  |  Zoom: {int(self.zoom_level * 100)}%")
            else:
                self.hud_coords_var.set(f"X: {x_mm:5.1f} mm  |  Y: {y_mm:5.1f} mm  (Off-sheet)")

    def _adjust_zoom(self, factor: float):
        self.zoom_level = max(0.3, min(4.0, self.zoom_level * factor))
        self._draw_page()

    def _fit_width(self):
        canv_w = self.canvas.winfo_width()
        if canv_w > 100:
            self.zoom_level = (canv_w - 60) / (self.cfg.page_width * 2.4)
            self.pan_x = 0.0
            self._draw_page()

    def _fit_page(self):
        canv_w = self.canvas.winfo_width()
        canv_h = self.canvas.winfo_height()
        if canv_w > 100 and canv_h > 100:
            scale_w = (canv_w - 60) / (self.cfg.page_width * 2.4)
            scale_h = (canv_h - 60) / (self.cfg.page_height * 2.4)
            self.zoom_level = min(scale_w, scale_h)
            self.pan_x = 0.0
            self.pan_y = 0.0
            self._draw_page()

    def _reset_view(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._draw_page()

    def _on_prev_page(self):
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            self.sim_stroke_idx = 0
            self._draw_page()

    def _on_next_page(self):
        if self.current_page_idx < len(self.pages) - 1:
            self.current_page_idx += 1
            self.sim_stroke_idx = 0
            self._draw_page()

    # --------------------------------------------------------------------------
    # Simulation / Drawing Animation Player
    # --------------------------------------------------------------------------
    def _toggle_simulation(self):
        """Starts or pauses virtual plotting simulation."""
        if self.sim_animating:
            self.sim_animating = False
            self.btn_sim.config(text="Simulate Plotting")
        else:
            if not self.pages:
                return
            self.sim_animating = True
            self.sim_stroke_idx = 0
            self.btn_sim.config(text="Pause Simulation")
            self._simulation_step()

    def _simulation_step(self):
        if not self.sim_animating or not self.pages:
            return

        current_page = self.pages[self.current_page_idx]
        total_strokes = len(current_page.strokes)

        self.sim_stroke_idx += self.sim_speed
        if self.sim_stroke_idx >= total_strokes:
            self.sim_stroke_idx = total_strokes
            self.sim_animating = False
            self.btn_sim.config(text="Simulate Plotting")
            self._draw_page()
            return

        self._draw_page()
        self.root.after(30, self._simulation_step)

    # --------------------------------------------------------------------------
    # G-Code Export & Serial Streaming
    # --------------------------------------------------------------------------
    def _on_export_gcode(self):
        if not self.pages:
            messagebox.showwarning("Empty", "No pages to export.")
            return

        out_dir = filedialog.askdirectory(title="Select G-Code Output Directory", initialdir="./gcode_output")
        if not out_dir:
            return

        os.makedirs(out_dir, exist_ok=True)
        compiler = gcode_compiler.GcodeCompiler(cfg=self.cfg)
        saved = []
        for p in self.pages:
            text, stats = compiler.compile_page(p)
            out_file = os.path.join(out_dir, f"page_{p.page_number}.gcode")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)
            saved.append(f"{os.path.basename(out_file)} ({stats.total_gcode_lines} lines)")

        messagebox.showinfo("Export Successful", f"Exported G-code files:\n" + "\n".join(saved))

    def _on_stream_plotter(self):
        if not self.pages:
            messagebox.showwarning("Empty", "No pages to plot.")
            return

        port = self.port_var.get().strip()
        dry_run = self.dry_run_var.get()
        baud = self.baud_var.get()

        if not dry_run and not port:
            messagebox.showerror("Port Required", "Please specify a valid USB serial port (e.g. /dev/ttyUSB0 or COM3).")
            return

        confirm = messagebox.askyesno(
            "Plotter Approval Check",
            f"Ready to plot {len(self.pages)} page(s) on {port if not dry_run else 'SIMULATOR'}?\n\n"
            f"Paper: ISO A4 (210 x 297 mm)\n"
            f"Simulation: {'ENABLED (Dry-Run)' if dry_run else 'DISABLED (Physical Plotter)'}\n\n"
            f"Proceed?"
        )
        if not confirm:
            return

        # Compile temporary files
        compiler = gcode_compiler.GcodeCompiler(cfg=self.cfg)
        temp_files = []
        for p in self.pages:
            text, _ = compiler.compile_page(p)
            tmp = f"/tmp/copycat_stream_p{p.page_number}.gcode"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            temp_files.append(tmp)

        def stream_worker():
            st = streamer.GrblStreamer(port=port, baud_rate=baud, dry_run=dry_run, cfg=self.cfg)
            if not st.connect():
                self.root.after(0, lambda: messagebox.showerror("Connection Error", f"Could not connect to {port}"))
                return

            try:
                for idx, fpath in enumerate(temp_files):
                    st.stream_file(fpath, page_number=idx + 1, total_pages=len(temp_files))
                self.root.after(0, lambda: messagebox.showinfo("Plot Complete", "All pages successfully plotted!"))
            finally:
                st.close()
                for fpath in temp_files:
                    if os.path.exists(fpath):
                        os.remove(fpath)

        threading.Thread(target=stream_worker, daemon=True).start()

    def _on_scan_ports(self):
        ports = streamer.GrblStreamer.list_available_ports()
        if ports:
            self.port_var.set(ports[0])
            messagebox.showinfo("Ports Found", f"Available ports:\n{', '.join(ports)}")
        else:
            messagebox.showinfo("No Devices", "No serial devices detected. Check USB connection or use Dry-Run mode.")


def launch_gui(filepath: str = "assignment.md"):
    """Entry point for Tkinter GUI Studio."""
    if not TKINTER_AVAILABLE:
        print("[ERROR] Tkinter is not available in this Python installation.")
        sys.exit(1)

    root = tk.Tk()
    app = CopyCatStudio(root, initial_file=filepath)
    root.mainloop()


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else "assignment.md"
    launch_gui(initial)
