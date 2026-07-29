"""
Terminal text formatting utilities.

Provides ANSI escape code styling with automatic TTY detection.
When output is not a terminal (piped to file, etc.), all formatting
is stripped so logs stay clean.
"""
import sys


def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class Style:
    """ANSI terminal formatting with TTY-aware fallback."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    RESET = "\033[0m"

    def __init__(self, enabled: bool | None = None):
        """
        Args:
            enabled: Force colors on/off. None = auto-detect TTY.
        """
        if enabled is None:
            enabled = _is_tty()
        if not enabled:
            self.BOLD = ""
            self.DIM = ""
            self.RED = ""
            self.GREEN = ""
            self.YELLOW = ""
            self.CYAN = ""
            self.RESET = ""

    # --- Phase headers ---

    def phase(self, title: str) -> str:
        bar = "═" * 60
        return (
            f"{self.CYAN}{bar}{self.RESET}\n"
            f"{self.BOLD}{self.CYAN}  {title}{self.RESET}\n"
            f"{self.CYAN}{bar}{self.RESET}"
        )

    # --- Fire headers ---

    def fire_header(self, index: int, total: int, fire_name: str, fire_year: int) -> str:
        return f"{self.BOLD}  [{index}/{total}] Processing: {fire_name} ({fire_year}){self.RESET}"

    # --- Step markers (├─ / └─) ---

    def step(self, number: int, total: int, label: str) -> str:
        connector = "└─" if number == total else "├─"
        return f"{self.CYAN}    {connector} [{number}/{total}] {label}{self.RESET}"

    # --- Sub-steps (dim with cyan │) ---

    def substep(self, label: str, last_step: bool = False) -> str:
        if last_step:
            return f"{self.CYAN}     {self.RESET}{self.DIM}    {label}{self.RESET}"
        return f"{self.CYAN}    │{self.RESET}{self.DIM}    {label}{self.RESET}"

    # --- Success / Error / Timing ---

    def success(self, fire_name: str, fire_year: int) -> str:
        return f"{self.BOLD}{self.GREEN}  ✓ {fire_name} ({fire_year}) complete.{self.RESET}"

    def error(self, fire_name: str, fire_year: int, message: str) -> str:
        return f"{self.BOLD}{self.RED}  ✗ {fire_name} ({fire_year}) FAILED: {message}{self.RESET}"

    def traceback(self, tb_text: str) -> str:
        return f"{self.DIM}{tb_text}{self.RESET}"

    def timing(self, index: int, total: int, elapsed_str: str, remaining_str: str) -> str:
        return f"{self.DIM}  [{index}/{total}] Elapsed: {elapsed_str} | Remaining: ~{remaining_str}{self.RESET}"
