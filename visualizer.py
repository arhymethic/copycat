"""
visualizer.py - Interactive Pygame Visual Inspection Canvas for Ruled Notebook Pages.

Displays:
1. Realistic A4 notebook paper (210 mm x 297 mm) with drop shadow.
2. Left margin rule (standard red/pink vertical line at 20 mm).
3. 8.0 mm horizontal ruled baselines (light notebook blue).
4. Page header margin (30 mm) and bottom cutoff margin (275 mm).
5. Real-time animated vector stroke trajectories rendered in dark ink.
6. Multi-page pagination controls (<- and -> arrow keys).
7. Interactive approval keys (Y to approve, N/ESC to cancel).
8. Headless fallback (exports PNG preview if display or Pygame is unavailable).
"""

import math
import os
import sys
import time
from typing import List, Tuple, Optional

import config
from layout_engine import Page, Stroke

# Try importing pygame safely
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Try importing Pillow for headless image export fallback
try:
    from PIL import Image, ImageDraw
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


class Visualizer:
    """Interactive notebook viewer built on Pygame with headless image fallback."""

    def __init__(self, pages: List[Page], cfg: Optional[config.PlotterConfig] = None):
        self.pages = pages or [Page(page_number=1)]
        self.cfg = cfg or config.DEFAULT_CONFIG
        self.current_page_idx = 0
        self.approval_decision: Optional[bool] = None

    def show(self, animate: bool = True) -> Optional[bool]:
        """
        Launches the preview. If GUI display is available, opens interactive
        Pygame window. If headless, exports PNG preview images.
        
        Returns:
            True if user explicitly approved (pressed 'Y'),
            False if user cancelled (pressed 'N' or ESC),
            None if window closed normally for console prompt.
        """
        # Check if Pygame and a graphical display environment exist
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform == "win32" or sys.platform == "darwin")
        
        if PYGAME_AVAILABLE and has_display:
            try:
                return self._run_pygame_loop(animate=animate)
            except Exception as e:
                print(f"[VISUALIZER] Pygame GUI initialization failed: {e}")
                print("[VISUALIZER] Falling back to headless image exporter.")

        # Headless fallback: Export PNG images
        self.export_preview_images()
        return None

    def export_preview_images(self, output_dir: str = "./preview_images") -> List[str]:
        """Renders pages to PNG files when running headless or for documentation."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []

        if not PILLOW_AVAILABLE:
            print("[VISUALIZER] Pillow not installed. Skipping PNG export.")
            return saved_paths

        # Render each page at 10 pixels per millimeter (2100 x 2970 px for crisp A4)
        px_per_mm = 8.0
        w_px = int(self.cfg.page_width * px_per_mm)
        h_px = int(self.cfg.page_height * px_per_mm)

        for page in self.pages:
            img = Image.new("RGB", (w_px, h_px), self.cfg.color_paper if hasattr(self.cfg, 'color_paper') else (252, 250, 246))
            draw = ImageDraw.Draw(img)

            # 1. Draw horizontal 8.0 mm ruled lines
            y_mm = self.cfg.margin_top
            while y_mm <= self.cfg.margin_bottom:
                py = int(y_mm * px_per_mm)
                draw.line([(0, py), (w_px, py)], fill=(195, 215, 238), width=1)
                y_mm += self.cfg.baseline_spacing

            # 2. Draw left margin line (pink/red)
            margin_x_px = int(self.cfg.margin_left * px_per_mm)
            draw.line([(margin_x_px, 0), (margin_x_px, h_px)], fill=(225, 95, 95), width=2)

            # 3. Draw strokes
            ink_color = (20, 36, 75)
            for stroke in page.strokes:
                if len(stroke) < 2:
                    continue
                pts = [(int(p[0] * px_per_mm), int(p[1] * px_per_mm)) for p in stroke]
                draw.line(pts, fill=ink_color, width=2)

            out_path = os.path.join(output_dir, f"preview_page_{page.page_number}.png")
            img.save(out_path)
            saved_paths.append(out_path)
            print(f"[VISUALIZER] Saved visual preview to: {out_path}")

        return saved_paths

    def _run_pygame_loop(self, animate: bool = True) -> Optional[bool]:
        """Runs the interactive Pygame window."""
        pygame.init()
        pygame.font.init()

        window_w = config.WINDOW_WIDTH
        window_h = config.WINDOW_HEIGHT
        screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
        pygame.display.set_caption("CopyCat - Pen Plotter Visual Inspection Pipeline")

        clock = pygame.time.Clock()
        font_title = pygame.font.SysFont("sans-serif", 18, bold=True)
        font_hud = pygame.font.SysFont("sans-serif", 14)
        font_legend = pygame.font.SysFont("monospace", 13)

        # Calculate paper layout inside the window
        hud_top_height = 45
        hud_bottom_height = 40
        avail_w = window_w - 60
        avail_h = window_h - hud_top_height - hud_bottom_height - 30

        scale_w = avail_w / self.cfg.page_width
        scale_h = avail_h / self.cfg.page_height
        scale = min(scale_w, scale_h)

        paper_w = int(self.cfg.page_width * scale)
        paper_h = int(self.cfg.page_height * scale)
        paper_x = (window_w - paper_w) // 2
        paper_y = hud_top_height + (avail_h - paper_h) // 2 + 10

        # Animation states
        animating = animate
        drawn_strokes_count = 0
        strokes_per_frame = 12

        running = True
        while running:
            dt = clock.tick(config.FPS)
            current_page = self.pages[self.current_page_idx]
            total_strokes = len(current_page.strokes)

            # Advance animation
            if animating:
                drawn_strokes_count = min(total_strokes, drawn_strokes_count + strokes_per_frame)
                if drawn_strokes_count >= total_strokes:
                    animating = False
            else:
                drawn_strokes_count = total_strokes

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.approval_decision = False
                        running = False
                    elif event.key == pygame.K_y:
                        self.approval_decision = True
                        running = False
                    elif event.key == pygame.K_n:
                        self.approval_decision = False
                        running = False
                    elif event.key in (pygame.K_RIGHT, pygame.K_n, pygame.K_DOWN):
                        if self.current_page_idx < len(self.pages) - 1:
                            self.current_page_idx += 1
                            animating = animate
                            drawn_strokes_count = 0
                    elif event.key in (pygame.K_LEFT, pygame.K_p, pygame.K_UP):
                        if self.current_page_idx > 0:
                            self.current_page_idx -= 1
                            animating = animate
                            drawn_strokes_count = 0
                    elif event.key == pygame.K_SPACE:
                        # Skip animation instantly
                        animating = False
                        drawn_strokes_count = total_strokes
                    elif event.key == pygame.K_r:
                        # Replay animation
                        animating = True
                        drawn_strokes_count = 0

            # ------------------------------------------------------------------
            # Render Frame
            # ------------------------------------------------------------------
            screen.fill(config.COLOR_DESK)

            # 1. Paper Drop Shadow
            shadow_rect = pygame.Rect(paper_x + 6, paper_y + 8, paper_w, paper_h)
            pygame.draw.rect(screen, config.COLOR_PAPER_SHADOW, shadow_rect, border_radius=4)

            # 2. Paper Surface
            paper_rect = pygame.Rect(paper_x, paper_y, paper_w, paper_h)
            pygame.draw.rect(screen, config.COLOR_PAPER, paper_rect, border_radius=2)
            pygame.draw.rect(screen, (220, 218, 210), paper_rect, width=1, border_radius=2)

            # 3. Horizontal 8.0 mm Ruled Baselines
            y_mm = self.cfg.margin_top
            while y_mm <= self.cfg.margin_bottom:
                py = paper_y + int(y_mm * scale)
                pygame.draw.line(screen, config.COLOR_RULED_LINE, (paper_x, py), (paper_x + paper_w, py), 1)
                y_mm += self.cfg.baseline_spacing

            # 4. Top Margin Accent Line (Double line or header rule)
            top_margin_py = paper_y + int(self.cfg.margin_top * scale)
            pygame.draw.line(screen, config.COLOR_HEADER_LINE, (paper_x, top_margin_py), (paper_x + paper_w, top_margin_py), 1)

            # 5. Left Vertical Margin Line (Notebook Pink/Red rule)
            left_margin_px = paper_x + int(self.cfg.margin_left * scale)
            pygame.draw.line(screen, config.COLOR_MARGIN_LINE, (left_margin_px, paper_y), (left_margin_px, paper_y + paper_h), 2)

            # 6. Right & Bottom Cutoff Boundary Guides (Subtle dashed guides)
            right_margin_px = paper_x + int(self.cfg.margin_right * scale)
            bottom_margin_py = paper_y + int(self.cfg.margin_bottom * scale)
            pygame.draw.line(screen, config.COLOR_CUTOFF_LINE, (right_margin_px, paper_y + int(self.cfg.margin_top * scale)), (right_margin_px, bottom_margin_py), 1)
            pygame.draw.line(screen, config.COLOR_CUTOFF_LINE, (left_margin_px, bottom_margin_py), (right_margin_px, bottom_margin_py), 1)

            # 7. Render Vector Strokes
            for s_idx in range(drawn_strokes_count):
                stroke = current_page.strokes[s_idx]
                if len(stroke) < 2:
                    continue
                screen_pts = [
                    (paper_x + int(px * scale), paper_y + int(py * scale))
                    for px, py in stroke
                ]
                pygame.draw.lines(screen, config.COLOR_INK, False, screen_pts, 2)

            # 8. Top HUD Banner
            hud_bg = pygame.Rect(0, 0, window_w, hud_top_height)
            pygame.draw.rect(screen, config.COLOR_HUD_BG, hud_bg)
            pygame.draw.line(screen, (50, 56, 70), (0, hud_top_height), (window_w, hud_top_height), 1)

            title_surf = font_title.render("CopyCat Inspection Preview", True, config.COLOR_HUD_TEXT)
            page_text = f"Page {self.current_page_idx + 1} of {len(self.pages)}  ({total_strokes} strokes)"
            page_surf = font_hud.render(page_text, True, config.COLOR_HUD_ACCENT)
            screen.blit(title_surf, (20, 12))
            screen.blit(page_surf, (window_w - page_surf.get_width() - 20, 15))

            # 9. Bottom Navigation & Action Bar
            bot_bg = pygame.Rect(0, window_h - hud_bottom_height, window_w, hud_bottom_height)
            pygame.draw.rect(screen, config.COLOR_HUD_BG, bot_bg)
            pygame.draw.line(screen, (50, 56, 70), (0, window_h - hud_bottom_height), (window_w, window_h - hud_bottom_height), 1)

            legend_text = "[<-/->] Pages   [Space] Skip Animation   [R] Replay   [Y] Approve Plot   [N/ESC] Abort"
            legend_surf = font_legend.render(legend_text, True, (180, 190, 205))
            screen.blit(legend_surf, (20, window_h - hud_bottom_height + 12))

            pygame.display.flip()

        pygame.quit()
        return self.approval_decision
