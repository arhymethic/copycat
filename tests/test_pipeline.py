"""
test_pipeline.py - Comprehensive Unit & Integration Tests for CopyCat.
"""

import os
import unittest
import tempfile
import config
import stroke_font
import layout_engine
import gcode_compiler
import streamer
import visualizer


class TestCopyCatPipeline(unittest.TestCase):

    def setUp(self):
        self.cfg = config.PlotterConfig()

    def test_config_dimensions(self):
        """Test default page dimensions and margins."""
        self.assertEqual(self.cfg.page_width, 210.0)
        self.assertEqual(self.cfg.page_height, 297.0)
        self.assertEqual(self.cfg.baseline_spacing, 8.0)
        self.assertEqual(self.cfg.margin_left, 20.0)
        self.assertEqual(self.cfg.margin_right, 190.0)
        self.assertEqual(self.cfg.margin_top, 30.0)
        self.assertEqual(self.cfg.margin_bottom, 275.0)
        self.assertEqual(self.cfg.z_up, 3.0)
        self.assertEqual(self.cfg.z_down, 0.0)

    def test_stroke_font_glyphs(self):
        """Test centerline glyph retrieval and word synthesis."""
        adv, strokes = stroke_font.get_glyph("A")
        self.assertGreater(adv, 0.0)
        self.assertGreater(len(strokes), 0)

        # Render word
        word_strokes, width = stroke_font.render_word_strokes(
            "Test", origin_x=20.0, baseline_y=38.0, scale=0.34,
            slant_deg=9.0, jitter_max=0.25, kerning_var=0.15, wobble=0.04
        )
        self.assertGreater(len(word_strokes), 0)
        self.assertGreater(width, 5.0)

    def test_markdown_parsing_and_layout(self):
        """Test layout engine Markdown parsing, classification, and styling."""
        engine = layout_engine.LayoutEngine(cfg=self.cfg)
        md_text = """# Assignment Title

Q1. Explain classical mechanics.
Ans: Mechanics is the study of motion and physical forces.
    F = m * a
"""
        pages = engine.parse_markdown(md_text)
        self.assertEqual(len(pages), 1)
        self.assertGreater(pages[0].total_stroke_count, 50)
        self.assertGreater(pages[0].line_count, 3)

    def test_multi_page_pagination(self):
        """Test auto-pagination when content exceeds the 275 mm bottom margin."""
        engine = layout_engine.LayoutEngine(cfg=self.cfg)
        long_md = "# Long Assignment\n\n"
        for i in range(1, 20):
            long_md += f"Q{i}. Question number {i} with a detailed explanation?\n"
            long_md += f"Ans: Answer number {i} expanding across multiple notebook lines to ensure page fills up.\n\n"

        pages = engine.parse_markdown(long_md)
        self.assertGreater(len(pages), 1, "Expected text to paginate across multiple pages")

    def test_gcode_compiler(self):
        """Test G-code generation with GRBL standard codes."""
        engine = layout_engine.LayoutEngine(cfg=self.cfg)
        pages = engine.parse_markdown("Q1. What is physics?\nAns: Natural science.")
        
        compiler = gcode_compiler.GcodeCompiler(cfg=self.cfg)
        gcode_text, stats = compiler.compile_page(pages[0])

        self.assertIn("G21", gcode_text)
        self.assertIn("G90", gcode_text)
        self.assertIn("G94", gcode_text)
        self.assertIn("G00", gcode_text)
        self.assertIn("G01", gcode_text)
        self.assertIn("M02", gcode_text)
        self.assertGreater(stats.draw_distance_mm, 0.0)
        self.assertGreater(stats.travel_distance_mm, 0.0)
        self.assertGreater(stats.total_gcode_lines, 20)

    def test_streamer_dry_run(self):
        """Test GrblStreamer in simulation/dry-run mode."""
        st = streamer.GrblStreamer(port="MOCK_TEST", dry_run=True, cfg=self.cfg)
        st.connect()
        self.assertTrue(st.ser.is_open)

        # Write sample gcode
        with tempfile.NamedTemporaryFile("w", suffix=".gcode", delete=False) as f:
            f.write("G21\nG90\nG00 Z3.000\nG00 X10.0 Y10.0\nG01 Z0.000\nG00 Z3.000\nM02\n")
            temp_path = f.name

        try:
            success = st.stream_file(temp_path, page_number=1, total_pages=1)
            self.assertTrue(success)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            st.close()

    def test_visualizer_headless_image_export(self):
        """Test that visualizer exports PNG preview in headless mode."""
        engine = layout_engine.LayoutEngine(cfg=self.cfg)
        pages = engine.parse_markdown("# Test Title\nQ1. Test question?\nAns: Test answer.")
        vis = visualizer.Visualizer(pages, cfg=self.cfg)

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = vis.export_preview_images(output_dir=tmp_dir)
            self.assertEqual(len(paths), 1)
            self.assertTrue(os.path.exists(paths[0]))
            self.assertGreater(os.path.getsize(paths[0]), 1000)


    def test_all_handwriting_styles(self):
        """Test all 24 handwriting presets across different font variants."""
        md = "# Style Check\nQ1. Test question?\nAns: Testing multiple styles."
        for style_name in config.STYLE_PRESETS:
            engine = layout_engine.LayoutEngine(cfg=self.cfg, style=style_name, engine_type="hershey")
            pages = engine.parse_markdown(md)
            self.assertGreater(pages[0].total_stroke_count, 10, f"Failed for style: {style_name}")
            preset = engine.get_style_preset(style_name)
            self.assertIn("slant", preset)
            self.assertIn("jitter", preset)

    def test_custom_style_overrides(self):
        """Test fine-tuning overrides for slant, jitter, wobble, kerning, and scale."""
        engine = layout_engine.LayoutEngine(
            cfg=self.cfg,
            style="natural",
            custom_slant=18.0,
            custom_jitter=0.45,
            custom_wobble=0.08,
            custom_kerning=0.25,
            custom_scale=0.42
        )
        preset = engine.get_style_preset("natural")
        self.assertEqual(preset["slant"], 18.0)
        self.assertEqual(preset["jitter"], 0.45)
        self.assertEqual(preset["wobble"], 0.08)
        self.assertEqual(preset["kerning_var"], 0.25)
        self.assertEqual(engine.base_scale, 0.42)

    def test_vasquez_cloned_repo_and_iam_styles(self):
        """Test that sjvasquez/handwriting-synthesis repo assets and drawing module are active."""
        import rnn_engine
        engine = rnn_engine.VASQUEZ_ENGINE
        self.assertTrue(engine.has_repo, "handwriting-synthesis repository should be cloned locally")
        self.assertTrue(engine.has_styles, "IAM styles folder should exist in cloned repo")
        self.assertTrue(engine.has_checkpoints, "Pretrained checkpoints should exist in cloned repo")

        # Test loading of raw IAM human handwriting strokes
        strokes, chars = engine.load_iam_style_raw(1)
        self.assertGreater(len(strokes), 0)
        self.assertGreater(len(chars), 0)

        # Verify drawing module integration
        self.assertTrue(rnn_engine.DRAWING_AVAILABLE, "drawing module from cloned repo should be loaded")
        import numpy as np
        # Test offset processing
        sample_offsets = strokes[:30].copy()
        polylines, width = engine._process_offsets_to_mm(sample_offsets, 20.0, 40.0, 4.8)
        self.assertIsInstance(polylines, list)
        self.assertGreater(width, 0.0)


    def test_ui_import(self):
        """Test that ui module is present and loads properly."""
        import ui
        self.assertTrue(hasattr(ui, "CopyCatStudio"))


if __name__ == "__main__":
    unittest.main()

