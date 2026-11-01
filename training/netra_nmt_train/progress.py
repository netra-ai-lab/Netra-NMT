# From the Netra Lab kit (templates/repo), generalised from Netra-OCR's
# training/netra_ocr_train/{utils.py,scripts/train.py}. Generic: use as is.
# See playbooks/TRAINING.md §4.
"""Progress tracking for a training run: bars, step lines, per-epoch summaries,
metrics.csv, early stopping and the final line.

    setup_logging(run_dir / "train.log")
    tracker = RunTracker(run_dir, epochs=args.epochs, metric="cer", patience=args.patience)
    with logging_redirect_tqdm():
        tracker.baseline(validate(model))                      # "Before training: ..."
        for epoch in range(start_epoch, args.epochs):
            bar = tracker.epoch_bar(epoch, total=len(train_set))
            for step, batch in enumerate(loader):
                ...
                bar.update(batch_size)
                bar.set_postfix(loss=f"{running / (step + 1):.4f}", lr=f"{lr:.1e}", refresh=False)
                tracker.step(epoch, step, steps_per_epoch, loss.item(), lr)
            bar.close()
            improved, stop = tracker.end_epoch(epoch, running / steps_per_epoch, validate(model), lr)
            save last.pth; if improved: save best.pth (weights only)
            if stop: break
    tracker.done(device)

Bars switch off by themselves when stdout isn't a terminal (nohup, a pipe,
CI); the step lines then go to the console too, else only to train.log.
"""

import csv
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm  # noqa: F401  (re-exported for train.py)

logger = logging.getLogger(__name__)


def setup_logging(log_file: str | Path | None = None):
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def file_only_logger(name: str = "progress") -> logging.Logger:
    """A logger that writes only to the log file, for the per-step lines a
    progress bar already shows on the screen."""
    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(logging.INFO)
    log.handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
    return log


def duration(seconds: float) -> str:
    if seconds < 60:
        return f"{max(1, round(seconds))} s"
    minutes = seconds / 60
    return f"{minutes:.1f} min" if minutes < 90 else f"{minutes / 60:.1f} h"


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


class RunTracker:
    """One training run's progress, summaries and metrics.csv.

    ``val`` dicts map metric names to fractions, e.g. {"cer": 0.012, "wer": 0.18,
    "exact": 0.84}; ``metric`` is the one that picks the best epoch.
    """

    def __init__(self, run_dir: str | Path, epochs: int, metric: str = "cer", lower_is_better: bool = True,
                 patience: int = 0, log_every: int = 50, unit: str = "sample"):
        self.run_dir, self.epochs, self.metric = Path(run_dir), epochs, metric
        self.lower, self.patience, self.log_every, self.unit = lower_is_better, patience, log_every, unit
        self.bars = sys.stderr.isatty()
        self.step_log = file_only_logger() if self.bars else logger
        self.best, self.best_epoch, self.stale = None, None, 0
        self.epoch_times, self.started, self.epoch_started = [], time.time(), None
        self.metrics_path = self.run_dir / "metrics.csv"

    def _better(self, value: float) -> bool:
        return self.best is None or (value < self.best if self.lower else value > self.best)

    def _scores(self, val: dict) -> str:
        return " ".join(f"{name} {pct(value)}" for name, value in val.items())

    def baseline(self, val: dict):
        logger.info(f"Before training: val {self._scores(val)}")

    def epoch_bar(self, epoch: int, total: int) -> tqdm:
        self.epoch_started = time.time()
        return tqdm(total=total, desc=f"epoch {epoch + 1}/{self.epochs}", unit=self.unit,
                    dynamic_ncols=True, leave=False, disable=None)

    def step(self, epoch: int, step: int, steps: int, loss: float, lr: float):
        if (step + 1) % self.log_every == 0 or step + 1 == steps:
            self.step_log.info(f"epoch {epoch + 1}/{self.epochs} step {step + 1}/{steps} loss={loss:.4f} lr={lr:.2e}")

    def end_epoch(self, epoch: int, loss: float, val: dict, lr: float) -> tuple[bool, bool]:
        """Log the summary line and append to metrics.csv; returns (improved, stop)."""
        seconds = time.time() - self.epoch_started
        self.epoch_times.append(seconds)
        improved = self._better(val[self.metric])
        if improved:
            self.best, self.best_epoch, self.stale = val[self.metric], epoch + 1, 0
        else:
            self.stale += 1
        left = (self.epochs - epoch - 1) * sum(self.epoch_times) / len(self.epoch_times)
        logger.info(f"epoch {epoch + 1}/{self.epochs}: loss {loss:.4f} · val {self._scores(val)}"
                    + (" (best)" if improved else f" (best {pct(self.best)})")
                    + f" · {duration(seconds)}" + (f" · about {duration(left)} left" if left else ""))

        new = not self.metrics_path.exists()
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["epoch", "train_loss", *[f"val_{name}" for name in val], "lr", "minutes"])
            writer.writerow([epoch + 1, f"{loss:.5f}", *[f"{value:.5f}" for value in val.values()],
                             f"{lr:.3e}", f"{seconds / 60:.2f}"])

        stop = bool(self.patience) and self.stale >= self.patience
        if stop:
            logger.info(f"No better validation {self.metric} for {self.stale} "
                        f"epoch{'s' if self.stale > 1 else ''}: stopping early.")
        return improved, stop

    def done(self, device=None):
        memory = ""
        if device is not None and getattr(device, "type", device) == "cuda":
            import torch
            memory = f", peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GB"
        best = f"Best validation {self.metric} {pct(self.best)} (epoch {self.best_epoch})" if self.best_epoch else "No epoch finished"
        logger.info(f"Done in {duration(time.time() - self.started)}{memory}. {best}: {self.run_dir / 'best.pth'}")
