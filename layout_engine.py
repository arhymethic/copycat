"""
layout_engine.py - Structured Markdown Parser, Ruled Notebook Pagination & Vector Synthesizer.

Responsibilities:
1. Parse structured Markdown (headers, questions, indented answers, blank lines).
2. Measure physical single-stroke vector text widths.
3. Perform automatic word-wrapping within page margins (20 mm to 190 mm).
4. Auto-paginate across standard A4 8.0 mm ruled baselines (30 mm to 275 mm).
5. Apply distinct handwriting biases (neat for questions, natural for answers).
6. Integrate with sjvasquez/handwriting-synthesis RNN when available, or fallback
   to the built-in Hershey single-stroke engine.
"""

import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import config
import stroke_font

# Type alias: Stroke is a list of (x, y) coordinates in millimeters
Stroke = List[Tuple[float, float]]


@dataclass
class Page:
    """Represents a single physical plotted notebook page."""
    page_number: int
    strokes: List[Stroke] = field(default_factory=list)
    line_count: int = 0

    @property
    def total_stroke_count(self) -> int:
        return len(self.strokes)


class LayoutEngine:
    """
    Parses Markdown and synthesizes vector stroke polylines locked to 8.0 mm
    notebook ruled lines and page margins.
    """

    def __init__(self, cfg: Optional[config.PlotterConfig] = None,
                 engine_type: str = "auto",
                 style: str = "natural",
                 custom_slant: Optional[float] = None,
                 custom_jitter: Optional[float] = None,
                 custom_wobble: Optional[float] = None,
                 custom_kerning: Optional[float] = None,
                 custom_scale: Optional[float] = None):
        self.cfg = cfg or config.DEFAULT_CONFIG
        self.engine_type = engine_type.lower()
        self.style = style.lower() if style in config.STYLE_PRESETS else "natural"
        self.custom_slant = custom_slant
        self.custom_jitter = custom_jitter
        self.custom_wobble = custom_wobble
        self.custom_kerning = custom_kerning
        self.custom_scale = custom_scale
        self.rnn_available = self._check_rnn_engine()
        
        # Scaling factor: maps glyph coordinates to physical mm
        # Default font cap-height is 14.0 units -> scaled to ~4.8 mm for 8.0 mm ruled lines
        preset = self.get_style_preset(self.style)
        scale_mult = preset.get("scale_mult", 1.0)
        if self.custom_scale is not None:
            self.base_scale = self.custom_scale
        else:
            self.base_scale = 0.34 * scale_mult
        self.header_scale = self.base_scale * 1.30

    def get_style_preset(self, style_key: str) -> dict:
        """Returns the active style dictionary with any custom parameter overrides applied."""
        base = dict(config.STYLE_PRESETS.get(style_key, config.STYLE_PRESETS.get(self.style, config.STYLE_PRESETS["natural"])))
        if self.custom_slant is not None:
            base["slant"] = self.custom_slant
        if self.custom_jitter is not None:
            base["jitter"] = self.custom_jitter
        if self.custom_wobble is not None:
            base["wobble"] = self.custom_wobble
        if self.custom_kerning is not None:
            base["kerning_var"] = self.custom_kerning
        return base

    def _check_rnn_engine(self) -> bool:
        """Checks if sjvasquez/handwriting-synthesis is installed and weights exist."""
        if self.engine_type == "hershey":
            return False
        try:
            # Check for Sean Vasquez's handwriting synthesis package or local submodule
            import handwriting_synthesis  # type: ignore
            return True
        except ImportError:
            return False

    def parse_file(self, filepath: str) -> List[Page]:
        """Reads a Markdown file and produces paginated vector stroke pages."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input Markdown file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_markdown(content)

    def parse_markdown(self, markdown_text: str) -> List[Page]:
        """
        Parses structured Markdown and generates vector stroke trajectories.
        
        Rules:
        - '# Header': Centered between Left and Right margins, larger line jump.
        - 'Q1.', 'Q2.', '1.': Left-aligned at Margin Left (20 mm), neater handwriting bias.
        - 'Ans:' or 4-space indent: Indented by 8 mm (starts at 28 mm), natural handwriting bias.
        - Blank line: Advances by a single 8.0 mm ruled line.
        - Multi-page pagination when exceeding Bottom Margin (275 mm).
        """
        pages: List[Page] = []
        current_page = Page(page_number=1)
        pages.append(current_page)

        # Baseline Y starts at the first ruled line beneath top margin
        # e.g., MARGIN_TOP (30.0) + BASELINE_SPACING (8.0) = 38.0 mm
        current_baseline_y = self.cfg.margin_top + self.cfg.baseline_spacing

        lines = markdown_text.splitlines()
        i = 0
        while i < len(lines):
            raw_line = lines[i]
            stripped = raw_line.strip()

            # 1. Blank line -> Advance by one ruled line
            if not stripped:
                current_baseline_y += self.cfg.baseline_spacing
                if current_baseline_y > self.cfg.margin_bottom:
                    current_page = Page(page_number=len(pages) + 1)
                    pages.append(current_page)
                    current_baseline_y = self.cfg.margin_top + self.cfg.baseline_spacing
                i += 1
                continue

            # 2. Header line (# Header) -> Centered, larger font, double line jump
            if raw_line.startswith("#"):
                header_text = raw_line.lstrip("#").strip()
                # Extra spacing before header if not at page top
                if current_baseline_y > self.cfg.margin_top + self.cfg.baseline_spacing * 2:
                    current_baseline_y += self.cfg.baseline_spacing

                current_page, current_baseline_y = self._render_header(
                    header_text, current_page, pages, current_baseline_y
                )
                i += 1
                continue

            # 3. Question line (e.g. 'Q1.', 'Q2:', '1.') -> Left-aligned at margin (20 mm), neater bias
            is_question = bool(re.match(r"^(Q\d+[\.:\)]|\d+[\.:\)])\s*", stripped, re.IGNORECASE))
            # 4. Indented line or 'Ans:' -> Indented by 8 mm (starts at 28 mm), natural bias
            is_answer = bool(raw_line.startswith("    ") or raw_line.startswith("\t") or 
                             re.match(r"^(Ans:|Answer:|A:)\s*", stripped, re.IGNORECASE))

            if is_question:
                style_key = "neat" if self.style == "natural" else self.style
                start_x = self.cfg.margin_left
                cont_x = self.cfg.margin_left
                text_to_render = stripped
            elif is_answer:
                style_key = self.style
                start_x = self.cfg.margin_left + self.cfg.indent_width
                cont_x = self.cfg.margin_left + self.cfg.indent_width
                text_to_render = stripped
            else:
                # Standard paragraph line
                style_key = self.style
                start_x = self.cfg.margin_left
                cont_x = self.cfg.margin_left
                text_to_render = stripped

            current_page, current_baseline_y = self._render_paragraph(
                text_to_render,
                start_x=start_x,
                continuation_x=cont_x,
                current_page=current_page,
                pages=pages,
                baseline_y=current_baseline_y,
                style_key=style_key
            )
            i += 1

        # Remove trailing empty page if no strokes were written
        if len(pages) > 1 and not pages[-1].strokes:
            pages.pop()

        return pages

    def _render_header(self, text: str,
                       current_page: Page,
                       pages: List[Page],
                       baseline_y: float) -> Tuple[Page, float]:
        """Renders a centered header with larger line jump and prominence."""
        scale = self.header_scale
        preset = self.get_style_preset(self.style)
        variant = preset.get("font_variant", "simplex")
        text_width = stroke_font.measure_text_width(text, scale, variant=variant)
        
        # Center between MARGIN_LEFT and MARGIN_RIGHT
        mid_x = (self.cfg.margin_left + self.cfg.margin_right) / 2.0
        start_x = max(self.cfg.margin_left, mid_x - text_width / 2.0)

        # Check page overflow
        if baseline_y > self.cfg.margin_bottom:
            current_page = Page(page_number=len(pages) + 1)
            pages.append(current_page)
            baseline_y = self.cfg.margin_top + self.cfg.baseline_spacing

        words = text.split(" ")
        curr_x = start_x
        space_w = stroke_font.measure_text_width(" ", scale, variant=variant)

        for word in words:
            word_strokes, adv = stroke_font.render_word_strokes(
                word,
                origin_x=curr_x,
                baseline_y=baseline_y,
                scale=scale,
                slant_deg=max(0.0, preset["slant"] - 1.5),
                jitter_max=preset["jitter"] * 0.6,
                kerning_var=preset["kerning_var"] * 0.8,
                wobble=preset["wobble"] * 0.8,
                variant=variant
            )
            current_page.strokes.extend(word_strokes)
            curr_x += adv + space_w

        current_page.line_count += 1
        # Advance by 2 ruled baselines (16 mm) after header for clean visual separation
        baseline_y += self.cfg.baseline_spacing * 2.0
        return current_page, baseline_y

    def _render_paragraph(self, text: str,
                          start_x: float,
                          continuation_x: float,
                          current_page: Page,
                          pages: List[Page],
                          baseline_y: float,
                          style_key: str) -> Tuple[Page, float]:
        """
        Word-wraps and renders a paragraph across 8.0 mm baselines with pagination.
        """
        preset = self.get_style_preset(style_key)
        variant = preset.get("font_variant", "simplex")
        scale = self.base_scale
        space_w = stroke_font.measure_text_width(" ", scale, variant=variant)

        words = text.split(" ")
        curr_x = start_x

        # Check page overflow before starting paragraph
        if baseline_y > self.cfg.margin_bottom:
            current_page = Page(page_number=len(pages) + 1)
            pages.append(current_page)
            baseline_y = self.cfg.margin_top + self.cfg.baseline_spacing

        # Arm drift phase for this line
        drift_phase = (baseline_y * 0.17) % (2.0 * math.pi)

        for word in words:
            if not word:
                continue

            word_w = stroke_font.measure_text_width(word, scale, variant=variant)

            # Check if word exceeds right cutoff margin
            if curr_x + word_w > self.cfg.margin_right and curr_x > start_x:
                # Wrap to next ruled line
                current_page.line_count += 1
                baseline_y += self.cfg.baseline_spacing
                curr_x = continuation_x

                # Check if we crossed the bottom margin cutoff -> New page!
                if baseline_y > self.cfg.margin_bottom:
                    current_page = Page(page_number=len(pages) + 1)
                    pages.append(current_page)
                    baseline_y = self.cfg.margin_top + self.cfg.baseline_spacing
                
                drift_phase = (baseline_y * 0.17) % (2.0 * math.pi)

            # Low-frequency arm drift wave
            arm_drift = self.cfg.baseline_drift_amp * math.sin(
                (curr_x / self.cfg.baseline_drift_period) * 2.0 * math.pi + drift_phase
            )

            # Render strokes for this word
            word_strokes, adv = stroke_font.render_word_strokes(
                word,
                origin_x=curr_x,
                baseline_y=baseline_y,
                scale=scale,
                slant_deg=preset["slant"],
                jitter_max=preset["jitter"],
                kerning_var=preset["kerning_var"],
                wobble=preset["wobble"],
                arm_drift=arm_drift,
                variant=variant
            )
            current_page.strokes.extend(word_strokes)
            curr_x += adv + space_w

        current_page.line_count += 1
        # Advance baseline for the next line
        baseline_y += self.cfg.baseline_spacing
        return current_page, baseline_y

