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


# Theme color palettes
THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "name": "Dark Midnight",
        "bg": "#14151f",
        "panel": "#1b1c28",
        "panel_alt": "#222434",
        "border": "#2c2e42",
        "text": "#e2e5f0",
        "muted": "#868ba3",
        "input": "#101119",
        "accent": "#4f86f7",
        "accent_hover": "#6b9bf9",
        "btn": "#252738",
        "btn_hover": "#32354c",
        "canvas_bg": "#0d0e14",
        "paper_bg": "#fbfaf6",
        "paper_border": "#2b2d3d",
        "ink": "#1a1f3c",
        "ruling": "#d8e5f2",
        "margin": "#e57373"
    },
    "oled": {
        "name": "OLED Charcoal",
        "bg": "#000000",
        "panel": "#0c0d10",
        "panel_alt": "#14161c",
        "border": "#1c1e26",
        "text": "#f0f2f8",
        "muted": "#71768e",
        "input": "#050608",
        "accent": "#00d2ff",
        "accent_hover": "#38deff",
        "btn": "#181a22",
        "btn_hover": "#252834",
        "canvas_bg": "#000000",
        "paper_bg": "#fcfbfa",
        "paper_border": "#222430",
        "ink": "#111425",
        "ruling": "#dbe3ed",
        "margin": "#ff5252"
    },
    "light": {
        "name": "Light Studio",
        "bg": "#f4f5f8",
        "panel": "#ffffff",
        "panel_alt": "#eef0f5",
        "border": "#d2d6e0",
        "text": "#1a1d28",
        "muted": "#6b7280",
        "input": "#ffffff",
        "accent": "#2563eb",
        "accent_hover": "#1d4ed8",
        "btn": "#e5e7eb",
        "btn_hover": "#d1d5db",
        "canvas_bg": "#e5e7eb",
        "paper_bg": "#ffffff",
        "paper_border": "#cbd5e1",
        "ink": "#1e293b",
        "ruling": "#e2e8f0",
        "margin": "#ef4444"
    },
    "sepia": {
        "name": "Warm Sepia",
        "bg": "#231f1d",
        "panel": "#2d2825",
        "panel_alt": "#38322e",
        "border": "#4a423d",
        "text": "#f3eee8",
        "muted": "#a89e94",
        "input": "#1d1a18",
        "accent": "#d97706",
        "accent_hover": "#f59e0b",
        "btn": "#3d3632",
        "btn_hover": "#4d4540",
        "canvas_bg": "#181514",
        "paper_bg": "#f7f1e5",
        "paper_border": "#4d4540",
        "ink": "#29211b",
        "ruling": "#e3d8c5",
        "margin": "#d96552"
    },
    "cyberpunk": {
        "name": "Matrix Terminal",
        "bg": "#0a0f0d",
        "panel": "#101815",
        "panel_alt": "#16221d",
        "border": "#1e332a",
        "text": "#e0f8ee",
        "muted": "#5a8c76",
        "input": "#070c0a",
        "accent": "#00ff88",
        "accent_hover": "#33ffa3",
        "btn": "#182a22",
        "btn_hover": "#223d31",
        "canvas_bg": "#050807",
        "paper_bg": "#f5f8f6",
        "paper_border": "#1e332a",
        "ink": "#0a1a12",
        "ruling": "#d2e5dc",
        "margin": "#ff4d6d"
    }
}


class InAppLogStream:
    """Mirrors console stdout stream to the in-app Tkinter log console safely."""
    def __init__(self, original_stream, callback):
        self.original_stream = original_stream
        self.callback = callback

    def write(self, s: str):
        self.original_stream.write(s)
        self.original_stream.flush()
        if s:
            self.callback(s)

    def flush(self):
        self.original_stream.flush()


class CopyCatStudio:
    """Professional Pen Plotter Desktop Studio for CopyCat."""

    def __init__(self, root: tk.Tk, initial_file: str = "assignment.md"):
        self.root = root
        self.root.title("CopyCat - Pen Plotter Studio")
        self.root.geometry("1300x880")
        self.root.minsize(1000, 720)

        # Active theme state
        self.active_theme_key = "dark"
        self.theme = THEMES[self.active_theme_key]
        self.root.configure(bg=self.theme["bg"])

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

        # Coordinate HUD & Log ticker
        self.hud_coords_var = tk.StringVar(value="X: 0.0 mm | Y: 0.0 mm")
        self.log_ticker_var = tk.StringVar(value="Ready")

        # Background threading state
        self._worker_thread: Optional[threading.Thread] = None
        self._update_pending: bool = False
        self._synthesis_active: bool = False

        # Hook stdout to in-app log viewer
        self._orig_stdout = sys.stdout
        self._log_stream = InAppLogStream(sys.stdout, self._append_log_message)
        sys.stdout = self._log_stream
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Build application
        self._setup_styles()
        self._load_file_from_disk()
        self._build_ui()
        self.update_pipeline()

        # Start background disk watcher
        self._schedule_file_watch()

    def _on_window_close(self):
        """Restores stdout and safely exits."""
        if hasattr(self, "_orig_stdout"):
            sys.stdout = self._orig_stdout
        self.root.destroy()

    def _setup_styles(self):
        """Initializes ttk theme engine and applies active theme."""
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self._apply_theme(self.active_theme_key)

    def _apply_theme(self, theme_key: str):
        """Dynamically applies color palette across all ttk and tk widgets."""
        if theme_key not in THEMES:
            theme_key = "dark"
        self.active_theme_key = theme_key
        t = THEMES[theme_key]
        self.theme = t

        # Update root and canvas
        self.root.configure(bg=t["bg"])
        if hasattr(self, "canvas"):
            self.canvas.configure(bg=t["canvas_bg"])
        if hasattr(self, "paned"):
            self.paned.configure(bg=t["border"])
        if hasattr(self, "editor_frame"):
            self.editor_frame.configure(bg=t["border"])
        if hasattr(self, "log_frame"):
            self.log_frame.configure(bg=t["border"])

        if hasattr(self, "editor_txt"):
            self.editor_txt.configure(
                bg=t["input"],
                fg=t["text"],
                insertbackground=t["accent"],
                selectbackground=t["accent"],
                selectforeground="#ffffff"
            )

        if hasattr(self, "log_txt"):
            self.log_txt.configure(
                bg=t["input"],
                fg=t["text"],
                insertbackground=t["accent"]
            )
            self._configure_log_tags()

        # TTK styles
        self.style.configure(".", background=t["bg"], foreground=t["text"], borderwidth=0)
        self.style.configure("TFrame", background=t["bg"])
        self.style.configure("Panel.TFrame", background=t["panel"])
        self.style.configure("PanelAlt.TFrame", background=t["panel_alt"])

        self.style.configure("TLabel", background=t["bg"], foreground=t["text"], font=("sans-serif", 9))
        self.style.configure("Panel.TLabel", background=t["panel"], foreground=t["text"], font=("sans-serif", 9))
        self.style.configure("Muted.TLabel", background=t["panel"], foreground=t["muted"], font=("sans-serif", 8))
        self.style.configure("Header.TLabel", background=t["panel"], foreground=t["text"], font=("sans-serif", 10, "bold"))
        self.style.configure("Accent.TLabel", background=t["panel"], foreground=t["accent"], font=("sans-serif", 9, "bold"))
        self.style.configure("Logo.TLabel", background=t["panel"], foreground=t["accent"], font=("sans-serif", 13, "bold"))

        self.style.configure("TButton",
                             background=t["btn"],
                             foreground=t["text"],
                             font=("sans-serif", 9),
                             padding=(8, 4),
                             borderwidth=1,
                             relief="flat")
        self.style.map("TButton",
                       background=[("active", t["btn_hover"]), ("disabled", t["bg"])],
                       foreground=[("disabled", t["muted"])])

        self.style.configure("Primary.TButton",
                             background=t["accent"],
                             foreground="#ffffff",
                             font=("sans-serif", 9, "bold"),
                             padding=(8, 5))
        self.style.map("Primary.TButton",
                       background=[("active", t["accent_hover"])])

        self.style.configure("TNotebook", background=t["panel"], borderwidth=0)
        self.style.configure("TNotebook.Tab",
                             background=t["panel_alt"],
                             foreground=t["muted"],
                             padding=(10, 5),
                             font=("sans-serif", 9))
        self.style.map("TNotebook.Tab",
                       background=[("selected", t["panel"])],
                       foreground=[("selected", t["text"])])

        self.style.configure("TCombobox",
                             fieldbackground=t["input"],
                             background=t["btn"],
                             foreground=t["text"],
                             selectbackground=t["accent"],
                             selectforeground="#ffffff",
                             font=("sans-serif", 9))

        self.style.configure("TEntry",
                             fieldbackground=t["input"],
                             foreground=t["text"])

        self.style.configure("TCheckbutton", background=t["panel"], foreground=t["text"])
        self.style.map("TCheckbutton", background=[("active", t["panel"])])

        self.style.configure("TScale", background=t["panel"], troughcolor=t["input"])

        self.style.configure("Horizontal.TProgressbar",
                             background=t["accent"],
                             troughcolor=t["input"],
                             bordercolor=t["border"],
                             lightcolor=t["accent"],
                             darkcolor=t["accent"])

        # Redraw canvas if pages exist
        if hasattr(self, "pages") and self.pages:
            self._draw_page()

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
        """Constructs split-pane interface with responsive layout."""
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.theme["border"], bd=1, sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True)
        self.paned = paned

        # ----------------------------------------------------------------------
        # Left Panel (Unified Studio Workspace - All controls & Editor visible)
        # ----------------------------------------------------------------------
        left_container = ttk.Frame(paned, style="Panel.TFrame", width=440)
        paned.add(left_container, minsize=380)

        # Header bar with Brand and Theme Switcher
        header_bar = ttk.Frame(left_container, style="Panel.TFrame", padding=(12, 10))
        header_bar.pack(fill=tk.X)

        title_lbl = ttk.Label(header_bar, text="COPYCAT", style="Logo.TLabel")
        title_lbl.pack(side=tk.LEFT)
        sub_lbl = ttk.Label(header_bar, text=" Studio", style="Muted.TLabel")
        sub_lbl.pack(side=tk.LEFT)

        theme_box = ttk.Frame(header_bar, style="Panel.TFrame")
        theme_box.pack(side=tk.RIGHT)
        ttk.Label(theme_box, text="Theme:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        self.theme_cb = ttk.Combobox(
            theme_box,
            values=[THEMES[k]["name"] for k in THEMES],
            state="readonly",
            width=13,
            font=("sans-serif", 8)
        )
        self.theme_cb.set(self.theme["name"])
        self.theme_cb.pack(side=tk.LEFT)
        self.theme_cb.bind("<<ComboboxSelected>>", self._on_theme_changed)

        # 1. Style & Parameters Card
        self._build_style_panel(left_container)

        # 2. Document Source Editor Card (Expands vertically)
        self._build_editor_panel(left_container)

        # 3. Machine & Action Bar (Bottom of sidebar)
        self._build_action_panel(left_container)

        # 4. Hidden / Popout Windows for Machine Settings and Logs
        self._build_logs_window()
        self._build_machine_window()

        # ----------------------------------------------------------------------
        # Right Panel (Interactive Canvas & Viewport)
        # ----------------------------------------------------------------------
        right_container = ttk.Frame(paned, style="TFrame")
        paned.add(right_container, minsize=500)

        self._build_canvas_view(right_container)

        # Auto-fit page on canvas startup
        self.root.after(150, self._fit_page)

    def _build_style_panel(self, parent: ttk.Frame):
        """Top card: Style dropdown, fine-tuning sliders, and instant parameter controls."""
        card = ttk.Frame(parent, style="Panel.TFrame", padding=(12, 4, 12, 6))
        card.pack(fill=tk.X)

        row_style = ttk.Frame(card, style="Panel.TFrame")
        row_style.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(row_style, text="Handwriting Style:", style="Header.TLabel").pack(side=tk.LEFT)

        btn_reset = ttk.Button(row_style, text="Reset", width=6, command=self._on_reset_sliders)
        btn_reset.pack(side=tk.RIGHT)

        style_choices = list(config.STYLE_PRESETS.keys())
        self.style_cb = ttk.Combobox(
            card,
            textvariable=self.style_var,
            values=style_choices,
            state="readonly",
            font=("sans-serif", 9, "bold")
        )
        self.style_cb.pack(fill=tk.X, pady=(2, 4))
        self.style_cb.bind("<<ComboboxSelected>>", self._on_style_selected)

        desc_lbl = ttk.Label(
            card,
            textvariable=self.style_desc_var,
            wraplength=400,
            style="Muted.TLabel"
        )
        desc_lbl.pack(fill=tk.X, pady=(0, 6))

        slider_box = ttk.Frame(card, style="Panel.TFrame")
        slider_box.pack(fill=tk.X, pady=(0, 4))

        self._create_slider(slider_box, "Slant Angle (°):", self.slant_var, -5.0, 32.0)
        self._create_slider(slider_box, "Text Scale:", self.scale_var, 0.20, 0.55)
        self._create_slider(slider_box, "Baseline Jitter (mm):", self.jitter_var, 0.0, 0.50)
        self._create_slider(slider_box, "Micro-Wobble / Tremor:", self.wobble_var, 0.0, 0.12)

    def _build_editor_panel(self, parent: ttk.Frame):
        """Middle card: Full Markdown editor taking available vertical space."""
        card = ttk.Frame(parent, style="Panel.TFrame", padding=(12, 4, 12, 4))
        card.pack(fill=tk.BOTH, expand=True)

        file_row = ttk.Frame(card, style="Panel.TFrame")
        file_row.pack(fill=tk.X, pady=(0, 4))

        self.file_label = ttk.Label(file_row, text=os.path.basename(self.current_filepath), style="Header.TLabel")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_browse = ttk.Button(file_row, text="Browse", width=7, command=self._on_browse_file)
        btn_browse.pack(side=tk.RIGHT, padx=(4, 0))

        btn_reload = ttk.Button(file_row, text="Reload", width=7, command=self._on_reload_disk)
        btn_reload.pack(side=tk.RIGHT)

        text_frame = tk.Frame(card, bg=self.theme["border"], bd=1)
        self.editor_frame = text_frame
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 6))

        self.editor_txt = tk.Text(
            text_frame,
            wrap=tk.WORD,
            bg=self.theme["input"],
            fg=self.theme["text"],
            insertbackground=self.theme["accent"],
            selectbackground=self.theme["accent"],
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

        action_bar = ttk.Frame(card, style="Panel.TFrame")
        action_bar.pack(fill=tk.X)

        btn_apply = ttk.Button(action_bar, text="Apply Changes & Re-render", style="Primary.TButton", command=self._on_apply_editor)
        btn_apply.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_save = ttk.Button(action_bar, text="Save Disk", width=9, command=self._on_save_editor_to_disk)
        btn_save.pack(side=tk.RIGHT)

    def _build_action_panel(self, parent: ttk.Frame):
        """Bottom card: Output actions, hardware communication, and console log viewer."""
        card = ttk.Frame(parent, style="Panel.TFrame", padding=(12, 4, 12, 10))
        card.pack(fill=tk.X)

        btn_row = ttk.Frame(card, style="Panel.TFrame")
        btn_row.pack(fill=tk.X, pady=(0, 4))

        btn_export = ttk.Button(btn_row, text="Export GRBL G-Code", command=self._on_export_gcode)
        btn_export.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_stream = ttk.Button(btn_row, text="Stream to Plotter", style="Primary.TButton", command=self._on_stream_plotter)
        btn_stream.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        tool_row = ttk.Frame(card, style="Panel.TFrame")
        tool_row.pack(fill=tk.X)

        btn_mach = ttk.Button(tool_row, text="Machine & Serial...", command=self._open_machine_dialog)
        btn_mach.pack(side=tk.LEFT)

        btn_logs = ttk.Button(tool_row, text="Generation Logs...", command=self._open_logs_window)
        btn_logs.pack(side=tk.RIGHT)

    def _build_logs_window(self):
        """Constructs dedicated generation logs popout window."""
        win = tk.Toplevel(self.root)
        win.title("CopyCat - Generation Logs Console")
        win.geometry("640x480")
        win.configure(bg=self.theme["panel"])
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.withdraw()
        self.logs_win = win

        top_row = ttk.Frame(win, style="Panel.TFrame", padding=(10, 8))
        top_row.pack(fill=tk.X)

        lbl = ttk.Label(top_row, text="Pipeline Logs", style="Header.TLabel")
        lbl.pack(side=tk.LEFT)

        btn_copy = ttk.Button(top_row, text="Copy", width=7, command=self._on_copy_logs)
        btn_copy.pack(side=tk.RIGHT, padx=(4, 0))

        btn_clear = ttk.Button(top_row, text="Clear", width=7, command=self._on_clear_logs)
        btn_clear.pack(side=tk.RIGHT)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        chk_scroll = ttk.Checkbutton(top_row, text="Auto-scroll", variable=self.auto_scroll_var)
        chk_scroll.pack(side=tk.RIGHT, padx=8)

        log_frame = tk.Frame(win, bg=self.theme["border"], bd=1)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log_frame = log_frame

        self.log_txt = tk.Text(
            self.log_frame,
            wrap=tk.WORD,
            bg=self.theme["input"],
            fg=self.theme["text"],
            insertbackground=self.theme["accent"],
            font=("monospace", 9),
            padx=8,
            pady=8,
            bd=0,
            state=tk.DISABLED
        )
        self.log_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_txt.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_txt.configure(yscrollcommand=scroll.set)

        self._configure_log_tags()

    def _open_logs_window(self):
        """Displays the generation logs popout console."""
        if hasattr(self, "logs_win"):
            self.logs_win.deiconify()
            self.logs_win.lift()
            self.logs_win.focus_set()

    def _show_logs_tab(self):
        """Shortcut from task bar to open logs console."""
        self._open_logs_window()

    def _build_machine_window(self):
        """Constructs dedicated machine settings popout dialog."""
        win = tk.Toplevel(self.root)
        win.title("CopyCat - Machine & Serial Settings")
        win.geometry("420x260")
        win.configure(bg=self.theme["panel"])
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.withdraw()
        self.machine_win = win

        content = ttk.Frame(win, style="Panel.TFrame", padding=16)
        content.pack(fill=tk.BOTH, expand=True)

        sec_lbl = ttk.Label(content, text="Plotter Hardware Communication", style="Header.TLabel")
        sec_lbl.pack(anchor=tk.W, pady=(0, 12))

        port_row = ttk.Frame(content, style="Panel.TFrame")
        port_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(port_row, text="Serial Port:", style="Panel.TLabel", width=12).pack(side=tk.LEFT)
        port_entry = ttk.Entry(port_row, textvariable=self.port_var, width=16)
        port_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(port_row, text="Scan Ports", command=self._on_scan_ports).pack(side=tk.LEFT)

        baud_row = ttk.Frame(content, style="Panel.TFrame")
        baud_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(baud_row, text="Baud Rate:", style="Panel.TLabel", width=12).pack(side=tk.LEFT)
        baud_cb = ttk.Combobox(baud_row, textvariable=self.baud_var, values=[115200, 250000, 57600], width=12, state="readonly")
        baud_cb.pack(side=tk.LEFT, padx=4)

        dry_chk = ttk.Checkbutton(content, text="Dry-Run Simulation Mode (No hardware required)", variable=self.dry_run_var)
        dry_chk.pack(anchor=tk.W, pady=(8, 16))

        btn_close = ttk.Button(content, text="Done", style="Primary.TButton", command=win.withdraw)
        btn_close.pack(fill=tk.X)

    def _open_machine_dialog(self):
        """Displays the machine settings dialog."""
        if hasattr(self, "machine_win"):
            self.machine_win.deiconify()
            self.machine_win.lift()
            self.machine_win.focus_set()

    def _on_theme_changed(self, event=None):
        """Callback when user selects a different theme from the combobox."""
        selected_name = self.theme_cb.get()
        for key, t_data in THEMES.items():
            if t_data["name"] == selected_name:
                self._apply_theme(key)
                print(f"[THEME] Switched theme to '{t_data['name']}'")
                break

    def _on_clear_logs(self):
        """Clears in-app log viewer."""
        if hasattr(self, "log_txt"):
            self.log_txt.configure(state=tk.NORMAL)
            self.log_txt.delete("1.0", tk.END)
            self.log_txt.configure(state=tk.DISABLED)

    def _on_copy_logs(self):
        """Copies all log contents to the system clipboard."""
        if hasattr(self, "log_txt"):
            content = self.log_txt.get("1.0", tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo("Copied", "Log contents copied to clipboard.")

    def _configure_log_tags(self):
        """Applies high-visibility syntax colors to log tags according to the active theme."""
        if not hasattr(self, "log_txt"):
            return
        is_light = (self.active_theme_key == "light")
        self.log_txt.tag_config("default", foreground=self.theme["text"])
        self.log_txt.tag_config("pipeline", foreground="#2563eb" if is_light else "#60a5fa", font=("monospace", 9, "bold"))
        self.log_txt.tag_config("layout", foreground="#b45309" if is_light else "#fbbf24")
        self.log_txt.tag_config("rnn", foreground="#7e22ce" if is_light else "#c084fc")
        self.log_txt.tag_config("synth", foreground="#047857" if is_light else "#34d399")
        self.log_txt.tag_config("cache", foreground="#4d7c0f" if is_light else "#a3e635")
        self.log_txt.tag_config("style", foreground="#ea580c" if is_light else "#fb923c")
        self.log_txt.tag_config("error", foreground="#dc2626" if is_light else "#f87171", font=("monospace", 9, "bold"))
        self.log_txt.tag_config("streamer", foreground="#0284c7" if is_light else "#38bdf8")
        self.log_txt.tag_config("disk", foreground="#475569" if is_light else "#94a3b8")

    def _append_log_message(self, s: str):
        """Thread-safe dispatch to append log text to in-app console."""
        if hasattr(self, "root") and self.root.winfo_exists():
            self.root.after_idle(lambda text=s: self._insert_log_text(text))

    def _insert_log_text(self, text: str):
        if not hasattr(self, "log_txt") or not self.log_txt.winfo_exists():
            return

        cleaned = text.strip()
        if cleaned:
            last_line = cleaned.splitlines()[-1].strip()
            if len(last_line) > 55:
                last_line = last_line[:52] + "..."
            self.log_ticker_var.set(last_line)

        self.log_txt.configure(state=tk.NORMAL)
        for line in text.splitlines(keepends=True):
            tag = "default"
            if "[PIPELINE]" in line:
                tag = "pipeline"
            elif "[LAYOUT]" in line:
                tag = "layout"
            elif "[RNN]" in line:
                tag = "rnn"
            elif "[SYNTH]" in line:
                tag = "synth"
            elif "[CACHE]" in line:
                tag = "cache"
            elif "[STYLE]" in line:
                tag = "style"
            elif "[ERROR]" in line or "Exception" in line or "Traceback" in line:
                tag = "error"
            elif "[STREAMER]" in line or "[P1/" in line:
                tag = "streamer"
            elif "[DISK]" in line or "[FILE]" in line:
                tag = "disk"

            self.log_txt.insert(tk.END, line, tag)

        if getattr(self, "auto_scroll_var", None) and self.auto_scroll_var.get():
            self.log_txt.see(tk.END)
        self.log_txt.configure(state=tk.DISABLED)

    def _build_canvas_view(self, parent: ttk.Frame):
        """Constructs interactive viewport toolbar, task bar, canvas, and status HUD."""
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

        # ----------------------------------------------------------------------
        # Generation Task Bar (Live synthesis progress, ETA, and logs shortcut)
        # ----------------------------------------------------------------------
        self.task_bar_frame = ttk.Frame(parent, style="PanelAlt.TFrame", padding=(10, 5))
        self.task_bar_frame.pack(fill=tk.X)

        self.task_status_lbl = ttk.Label(self.task_bar_frame, text="Ready", style="Panel.TLabel")
        self.task_status_lbl.pack(side=tk.LEFT, padx=(0, 8))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self.task_bar_frame,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
            length=180
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.task_pct_lbl = ttk.Label(self.task_bar_frame, text="0%", style="Accent.TLabel", width=5)
        self.task_pct_lbl.pack(side=tk.LEFT, padx=(0, 6))

        self.task_time_lbl = ttk.Label(self.task_bar_frame, text="", style="Muted.TLabel", width=8)
        self.task_time_lbl.pack(side=tk.LEFT, padx=(0, 8))

        btn_view_logs = ttk.Button(self.task_bar_frame, text="View Logs", command=self._show_logs_tab)
        btn_view_logs.pack(side=tk.RIGHT)

        # Drawing Canvas
        self.canvas = tk.Canvas(parent, bg=self.theme["canvas_bg"], highlightthickness=0, cursor="fleur")
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

        # Bottom Status HUD & Live Log Ticker
        status_hud = ttk.Frame(parent, style="Panel.TFrame", padding=(10, 4))
        status_hud.pack(fill=tk.X)

        self.hud_label = ttk.Label(status_hud, textvariable=self.hud_coords_var, style="Muted.TLabel")
        self.hud_label.pack(side=tk.LEFT)

        ttk.Separator(status_hud, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.log_ticker = ttk.Label(status_hud, textvariable=self.log_ticker_var, style="Accent.TLabel")
        self.log_ticker.pack(side=tk.LEFT)

        guide_lbl = ttk.Label(status_hud, text="Drag to Pan | Scroll to Zoom | Click Simulate to preview", style="Muted.TLabel")
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
            if hasattr(self, "_slider_timer") and self._slider_timer:
                self.root.after_cancel(self._slider_timer)
            self._slider_timer = self.root.after(300, self.update_pipeline)

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
            engine_name = preset.get("engine", "hershey")
            style_id = preset.get("rnn_style_id", "N/A")
            print(f"\n[STYLE] Preset selected: '{key}' ({preset.get('name', key)})")
            print(f"        Engine: {engine_name} | Style ID: {style_id}")
            print(f"        Active parameters: Slant={self.slant_var.get():.2f}, Jitter={self.jitter_var.get():.2f}, Scale={self.scale_var.get():.2f}")
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
            print(f"[STYLE] Reset parameters to default values for '{key}'.")
            self.update_pipeline()

    def _on_apply_editor(self):
        """Applies text from the embedded editor, updates state, and re-renders."""
        self.markdown_content = self.editor_txt.get("1.0", tk.END).strip()
        print(f"\n[EDITOR] Applying changes from editor ({len(self.markdown_content.splitlines())} lines)...")
        self.update_pipeline()

    def _on_save_editor_to_disk(self):
        """Saves current editor content to disk."""
        self.markdown_content = self.editor_txt.get("1.0", tk.END).strip()
        try:
            with open(self.current_filepath, "w", encoding="utf-8") as f:
                f.write(self.markdown_content)
            self.last_mtime = os.path.getmtime(self.current_filepath)
            print(f"[DISK] Successfully saved to {self.current_filepath}")
            messagebox.showinfo("Saved", f"Successfully saved to {os.path.basename(self.current_filepath)}")
        except Exception as e:
            print(f"[ERROR] Could not save to disk: {e}", file=sys.stderr)
            messagebox.showerror("Save Error", f"Could not write file: {e}")

    def _on_reload_disk(self):
        """Explicitly reloads assignment content from disk."""
        print(f"[DISK] Reloading contents from {self.current_filepath}...")
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
            print(f"[FILE] Opened file: {self.current_filepath}")
            self._load_file_from_disk()
            self.editor_txt.delete("1.0", tk.END)
            self.editor_txt.insert("1.0", self.markdown_content)
            self.update_pipeline()

    def update_pipeline(self):
        """Dispatches vector synthesis pipeline to a background worker thread."""
        # Sync editor content if available
        if hasattr(self, "editor_txt"):
            txt_val = self.editor_txt.get("1.0", tk.END).strip()
            if txt_val:
                self.markdown_content = txt_val

        # If a worker thread is currently running, queue as pending
        if self._worker_thread and self._worker_thread.is_alive():
            self._update_pending = True
            print("[PIPELINE] Generation in progress; update request queued as pending.")
            return

        self._update_pending = False
        self.stats_label.config(text="Synthesizing...")
        self.hud_coords_var.set("Synthesizing handwriting strokes... Please wait.")

        self._worker_thread = threading.Thread(target=self._synthesis_worker, daemon=True)
        self._worker_thread.start()

    def _synthesis_worker(self):
        """Background thread worker for non-blocking stroke synthesis."""
        t_start = time.time()
        self._synthesis_active = True
        filename = os.path.basename(self.current_filepath)
        print(f"\n[PIPELINE] Starting vector synthesis for '{filename}'...")

        def _update_start():
            if hasattr(self, "progress_var"):
                self.progress_var.set(0.0)
            if hasattr(self, "task_pct_lbl"):
                self.task_pct_lbl.config(text="0%")
            if hasattr(self, "task_status_lbl"):
                self.task_status_lbl.config(text=f"Synthesizing '{filename}'...")
            if hasattr(self, "task_time_lbl"):
                self.task_time_lbl.config(text="0.0s")
        self.root.after(0, _update_start)

        def _timer_pulse():
            if getattr(self, "_synthesis_active", False):
                elapsed = time.time() - t_start
                if hasattr(self, "task_time_lbl"):
                    self.task_time_lbl.config(text=f"{elapsed:.1f}s")
                self.root.after(100, _timer_pulse)
        self.root.after(100, _timer_pulse)

        layout = layout_engine.LayoutEngine(
            cfg=self.cfg,
            style=self.style_var.get(),
            custom_slant=self.slant_var.get(),
            custom_jitter=self.jitter_var.get(),
            custom_wobble=self.wobble_var.get(),
            custom_kerning=self.kerning_var.get(),
            custom_scale=self.scale_var.get()
        )

        def progress_cb(current: int, total: int, text: str):
            disp = text[:24] + ("..." if len(text) > 24 else "")
            pct = min(100.0, (current / max(1, total)) * 100.0)
            def _update_ui_progress():
                if hasattr(self, "progress_var"):
                    self.progress_var.set(pct)
                if hasattr(self, "task_pct_lbl"):
                    self.task_pct_lbl.config(text=f"{int(pct)}%")
                if hasattr(self, "task_status_lbl"):
                    self.task_status_lbl.config(text=f"Line {current}/{total}: \"{disp}\"")
                self.hud_coords_var.set(f"Synthesizing ({current}/{total}): \"{disp}\"")
                self.stats_label.config(text=f"Synthesizing ({current}/{total})")
            self.root.after(0, _update_ui_progress)

        try:
            pages = layout.parse_markdown(self.markdown_content, progress_callback=progress_cb)
            self._synthesis_active = False
            elapsed = time.time() - t_start
            print(f"[PIPELINE] Synthesis finished in {elapsed:.2f}s ({len(pages)} page(s)).")
            self.root.after(0, lambda: self._on_synthesis_success(pages, elapsed))
        except Exception as e:
            self._synthesis_active = False
            print(f"[ERROR] Synthesis failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._on_synthesis_error(str(e)))

    def _on_synthesis_success(self, pages: List[layout_engine.Page], elapsed: float = 0.0):
        """Main thread callback to refresh view with newly generated pages."""
        self.pages = pages
        if self.current_page_idx >= len(self.pages):
            self.current_page_idx = max(0, len(self.pages) - 1)

        self.sim_stroke_idx = 0
        self._draw_page()

        total_strokes = sum(len(p.strokes) for p in pages)
        if hasattr(self, "progress_var"):
            self.progress_var.set(100.0)
        if hasattr(self, "task_pct_lbl"):
            self.task_pct_lbl.config(text="100%")
        if hasattr(self, "task_status_lbl"):
            self.task_status_lbl.config(text=f"Ready - Generated {len(pages)} page(s) ({total_strokes} strokes)")
        if hasattr(self, "task_time_lbl"):
            self.task_time_lbl.config(text=f"{elapsed:.1f}s")

        self.hud_coords_var.set("Ready | Drag to Pan | Scroll to Zoom")
        print(f"[GUI] Canvas updated: Displaying Page {self.current_page_idx + 1} of {len(self.pages)}")

        # Check if another update was requested while worker was running
        if self._update_pending:
            self._update_pending = False
            print("[PIPELINE] Executing queued pending update...")
            self.root.after(50, self.update_pipeline)

    def _on_synthesis_error(self, err_msg: str):
        """Main thread callback when layout parsing fails."""
        if hasattr(self, "progress_var"):
            self.progress_var.set(0.0)
        if hasattr(self, "task_pct_lbl"):
            self.task_pct_lbl.config(text="Error")
        if hasattr(self, "task_status_lbl"):
            self.task_status_lbl.config(text=f"Failed: {err_msg[:30]}")
        self.stats_label.config(text="Error")
        self.hud_coords_var.set("Error during synthesis")
        messagebox.showerror("Parsing Error", f"Failed to layout document:\n{err_msg}")

    def _draw_page(self):
        """Draws realistic notebook paper, margins, and single-stroke trajectories."""
        self.canvas.delete("all")
        if not self.pages:
            return

        current_page = self.pages[self.current_page_idx]
        total_pages = len(self.pages)
        t = self.theme

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
        center_y = max(15.0, (canv_h - paper_h) / 2.0) + self.pan_y

        self._paper_origin_x = center_x
        self._paper_origin_y = center_y
        self._px_per_mm = base_px_per_mm

        # 1. Drop shadow
        shadow_fill = "#000000" if self.active_theme_key != "light" else "#cbd5e1"
        self.canvas.create_rectangle(
            center_x + 8, center_y + 10, center_x + paper_w + 8, center_y + paper_h + 10,
            fill=shadow_fill, outline=""
        )

        # 2. Paper Sheet (Notebook sheet)
        self.canvas.create_rectangle(
            center_x, center_y, center_x + paper_w, center_y + paper_h,
            fill=t["paper_bg"], outline=t["paper_border"], width=1
        )

        # 3. Horizontal 8.0 mm Ruled Notebook Lines
        y_mm = self.cfg.margin_top
        while y_mm <= self.cfg.margin_bottom + 5.0:
            py = center_y + y_mm * base_px_per_mm
            self.canvas.create_line(
                center_x, py, center_x + paper_w, py,
                fill=t["ruling"], width=1
            )
            y_mm += self.cfg.baseline_spacing

        # 4. Vertical Margin Line
        margin_x = center_x + self.cfg.margin_left * base_px_per_mm
        self.canvas.create_line(
            margin_x, center_y, margin_x, center_y + paper_h,
            fill=t["margin"], width=1
        )

        # 5. Right Margin Boundary Guideline (Dotted line at 190 mm)
        cutoff_x = center_x + self.cfg.margin_right * base_px_per_mm
        guide_col = "#334155" if self.active_theme_key != "light" else "#94a3b8"
        self.canvas.create_line(
            cutoff_x, center_y + self.cfg.margin_top * base_px_per_mm,
            cutoff_x, center_y + self.cfg.margin_bottom * base_px_per_mm,
            fill=guide_col, dash=(2, 4)
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
                fill=t["ink"],
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
