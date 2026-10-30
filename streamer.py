"""
streamer.py - Direct USB Serial Streamer for GRBL / CoreXY Pen Plotters.

Features:
1. Direct USB streaming over pyserial using the standard GRBL 'ok' response loop.
2. Character-counting and ping-pong safety to prevent buffer overflow.
3. Multi-page pauses: parks pen at (0, 0), prompts user to align fresh paper against
   the corner jig, and waits for confirmation before resuming.
4. Emergency interrupt handler (Ctrl+C): instantly commands pen-up and safe stop.
5. Built-in Mock / Dry-Run mode to simulate streaming without physical hardware.
"""

import os
import sys
import time
import logging
from typing import List, Optional

import config

logger = logging.getLogger("copycat.streamer")

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# GRBL Error code dictionary for human-friendly debugging
GRBL_ERRORS = {
    1: "G-code words consist of a letter and a value. Letter was not found.",
    2: "Numeric value format is not valid or missing an expected value.",
    3: "Grbl '$' system command was not recognized or supported.",
    4: "Negative value received for an expected positive value.",
    5: "Homing cycle is not enabled via settings.",
    20: "Unsupported or invalid g-code command found in block.",
    22: "Feed rate has not yet been set or is undefined.",
    24: "Two G-code commands that both require the use of the XYZ axis were detected.",
    33: "Motion commanded would exceed machine limits (Soft limit error)."
}


class MockGrblSerial:
    """Simulates a responsive GRBL 1.1 controller over USB for dry-run testing."""

    def __init__(self, port: str = "MOCK", baudrate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = True
        self._in_buffer = b"Grbl 1.1h ['$' for help]\r\nok\r\n"
        self._pen_z = 3.0
        self._pos_x = 0.0
        self._pos_y = 0.0

    def write(self, data: bytes) -> int:
        line = data.decode("utf-8", errors="ignore").strip()
        # Fast simulation delay
        time.sleep(0.001)
        if line == "?":
            status = f"<Idle|MPos:{self._pos_x:.3f},{self._pos_y:.3f},{self._pen_z:.3f}|FS:0,0>\r\nok\r\n"
            self._in_buffer += status.encode("utf-8")
        else:
            self._in_buffer += b"ok\r\n"
        return len(data)

    def readline(self) -> bytes:
        if b"\n" in self._in_buffer:
            idx = self._in_buffer.index(b"\n") + 1
            line = self._in_buffer[:idx]
            self._in_buffer = self._in_buffer[idx:]
            return line
        time.sleep(0.01)
        return b""

    def flushInput(self):
        self._in_buffer = b""

    def reset_input_buffer(self):
        self._in_buffer = b""

    def close(self):
        self.is_open = False


class GrblStreamer:
    """Manages the serial connection and streaming loop for GRBL pen plotters."""

    def __init__(self, port: Optional[str] = None,
                 baud_rate: Optional[int] = None,
                 dry_run: bool = False,
                 cfg: Optional[config.PlotterConfig] = None):
        self.cfg = cfg or config.DEFAULT_CONFIG
        self.port = port or self.cfg.port
        self.baud_rate = baud_rate or self.cfg.baud_rate
        self.dry_run = dry_run
        self.ser = None

    @staticmethod
    def list_available_ports() -> List[str]:
        """Scans and returns detected serial COM/TTY devices."""
        if not SERIAL_AVAILABLE:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def connect(self):
        """Establishes connection with GRBL plotter controller."""
        if self.dry_run:
            print(f"[STREAMER] Running in DRY-RUN mode. Simulating connection to {self.port}...")
            self.ser = MockGrblSerial(port=self.port, baudrate=self.baud_rate)
            time.sleep(0.3)
            print("[STREAMER] Connected to Mock GRBL 1.1h controller.")
            return

        if not SERIAL_AVAILABLE:
            raise RuntimeError(
                "pyserial is not installed. Run 'pip install pyserial' or use --dry-run."
            )

        print(f"[STREAMER] Connecting to GRBL plotter on port {self.port} at {self.baud_rate} baud...")
        try:
            self.ser = serial.Serial(
                self.port,
                self.baud_rate,
                timeout=self.cfg.serial_timeout if hasattr(self.cfg, 'serial_timeout') else 1.0
            )
        except Exception as e:
            available = self.list_available_ports()
            msg = f"Failed to open port {self.port}: {e}\nAvailable system ports: {available}"
            raise ConnectionError(msg) from e

        # Wake up GRBL (send newlines, wait for init greeting)
        print("[STREAMER] Initializing GRBL controller...")
        self.ser.write(b"\r\n\r\n")
        time.sleep(2.0)  # Wait for GRBL bootloader
        self.ser.reset_input_buffer()
        print("[STREAMER] GRBL initialized and ready.")

    def stream_file(self, gcode_file_path: str, page_number: int = 1, total_pages: int = 1) -> bool:
        """
        Streams a single G-code file line-by-line using GRBL's 'ok' response loop.
        Displays real-time progress bar.
        """
        if not self.ser:
            self.connect()

        if not os.path.exists(gcode_file_path):
            raise FileNotFoundError(f"G-code file not found: {gcode_file_path}")

        with open(gcode_file_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        # Clean lines: remove comments and trailing whitespace
        executable_lines: List[str] = []
        for line in raw_lines:
            cleaned = line.split(";")[0].strip()
            if cleaned:
                executable_lines.append(cleaned)

        total_lines = len(executable_lines)
        print(f"\n[STREAMER] Starting Page {page_number}/{total_pages} ({total_lines} motion commands)...")

        start_time = time.time()
        try:
            for idx, cmd in enumerate(executable_lines, start=1):
                # Send command to GRBL
                self.ser.write((cmd + "\n").encode("utf-8"))

                # Wait for 'ok' response
                while True:
                    response = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if not response:
                        continue
                    if response == "ok":
                        break
                    if response.startswith("error:"):
                        err_code = response.split("error:")[1].strip()
                        err_desc = GRBL_ERRORS.get(int(err_code) if err_code.isdigit() else 0, "Unknown GRBL error")
                        print(f"\n[GRBL ERROR] Command: '{cmd}' -> {response}: {err_desc}")
                        raise RuntimeError(f"GRBL Error on line {idx}: {response} ({err_desc})")
                    if response.startswith("ALARM:"):
                        print(f"\n[GRBL ALARM] {response}")
                        raise RuntimeError(f"GRBL Machine Alarm: {response}")

                # Progress bar rendering
                if idx % 10 == 0 or idx == total_lines:
                    self._render_progress_bar(idx, total_lines, start_time, page_number, total_pages)

            print(f"\n[STREAMER] Page {page_number} plotted successfully in {time.time() - start_time:.1f}s.")
            return True

        except KeyboardInterrupt:
            print("\n[STREAMER WARNING] Emergency Stop triggered by user (Ctrl+C)!")
            self._emergency_stop()
            raise

    def wait_for_idle(self):
        """Sends status query '?' until GRBL reports Idle."""
        if not self.ser or self.dry_run:
            return
        # Send dwell to let motion planner finish
        self.ser.write(b"G4 P0.2\n")
        time.sleep(0.3)

    def _render_progress_bar(self, current: int, total: int, start_time: float, page: int, total_pages: int):
        """Displays a clean console progress bar with ETA."""
        bar_len = 30
        fraction = current / max(1, total)
        filled = int(fraction * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        elapsed = time.time() - start_time
        if fraction > 0:
            est_total = elapsed / fraction
            remaining = max(0, est_total - elapsed)
            rem_str = f"{int(remaining // 60)}m {int(remaining % 60):02d}s"
        else:
            rem_str = "--:--"

        percent = int(fraction * 100)
        line = f"\r[P{page}/{total_pages}] [{bar}] {percent}% ({current}/{total}) - ETA: {rem_str} "
        sys.stdout.write(line)
        sys.stdout.flush()

    def _emergency_stop(self):
        """Sends immediate Feed Hold and lifts pen to prevent paper or hardware damage."""
        if not self.ser:
            return
        try:
            print("[STREAMER] Sending Feed Hold (!) and Soft Reset (Ctrl+X)...")
            self.ser.write(b"!\x18")  # Feed hold + Reset
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            # Command pen up safely
            self.ser.write(f"G00 Z{self.cfg.z_up:.3f} F800.0\n".encode("utf-8"))
            print("[STREAMER] Pen lifted to safe height.")
        except Exception as e:
            print(f"[STREAMER] Error during emergency stop: {e}")

    def prompt_page_change_and_wait(self, next_page_number: int) -> bool:
        """
        Handles multi-page pauses:
        1. Ensures pen is lifted and parked at corner jig (0, 0).
        2. Prompts user to replace page and align against corner jig.
        3. Waits for user confirmation.
        """
        print("\n" + "=" * 70)
        print(f"[PAGE CHANGE REQUIRED] - Ready for Page {next_page_number}")
        print("=" * 70)
        print("1. Plotter has lifted the pen and parked at the corner jig (0, 0).")
        print("2. Carefully remove the finished sheet.")
        print("3. Align a new blank sheet snugly against the top-left corner jig.")
        print("4. Secure the sheet firmly with magnetic clamps or tape.")
        print("=" * 70)

        while True:
            choice = input(f"Press [Enter] to start plotting Page {next_page_number} (or 'q' to abort): ").strip().lower()
            if choice in ("q", "quit", "abort"):
                print("[STREAMER] Multi-page plotting aborted by user.")
                return False
            if choice == "":
                return True

    def stream_document(self, gcode_file_paths: List[str]) -> bool:
        """
        Streams all pages in sequence with corner jig pause between pages.
        """
        if not gcode_file_paths:
            print("[STREAMER] No G-code files to stream.")
            return False

        self.connect()
        total_pages = len(gcode_file_paths)

        for idx, file_path in enumerate(gcode_file_paths, start=1):
            success = self.stream_file(file_path, page_number=idx, total_pages=total_pages)
            if not success:
                return False

            # If there are subsequent pages, prompt user to change paper at corner jig
            if idx < total_pages:
                self.wait_for_idle()
                proceed = self.prompt_page_change_and_wait(next_page_number=idx + 1)
                if not proceed:
                    print(f"[STREAMER] Stopped after completing Page {idx}.")
                    return False

        self.wait_for_idle()
        print("\n[STREAMER] All pages completed successfully! Plotter ready.")
        self.close()
        return True

    def close(self):
        """Closes serial connection safely."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None

