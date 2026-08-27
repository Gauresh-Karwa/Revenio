from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, total: int, label: str = "", width: int = 30) -> None:
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self.start_time = time.time()
        self.current = 0

    def update(self, step: int = 1, suffix: str = "") -> None:
        self.current += step
        self._render(suffix)

    def _render(self, suffix: str = "") -> None:
        frac = min(self.current / self.total, 1.0)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start_time
        line = f"\r{self.label} [{bar}] {self.current}/{self.total} ({frac*100:5.1f}%) {elapsed:5.1f}s {suffix}"
        sys.stdout.write(line)
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()