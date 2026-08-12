"""GPU clock and power state, recorded next to every latency row.

WHY THIS MODULE EXISTS
-----------------------
Three speed tables were published from three machines and none of them recorded
the state of the GPU that produced them. That turned out to matter more than the
GPU model: on the same benchmark, an A100 MIG 3g.40gb slice (42 SM) came out
~3.3x FASTER than a full A100-PCIE-40GB (108 SM), near-uniformly across five
architectures with unrelated FLOP profiles. A near-constant multiplier across
unrelated architectures is not a compute result; it is fixed per-call overhead
and clock state. The leading suspect was `persistence_mode: Disabled` with an
applications clock of 765 MHz against a 1410 max -- but it stayed a suspect,
because nobody had sampled the clock UNDER LOAD.

So: `probe()` for the static configuration, and `ClockSampler` for what the
clock actually did while the timing loop ran. A latency number without both is
a number about a machine on a particular afternoon, not about a model.

EVERYTHING HERE DEGRADES TO None
---------------------------------
`nvidia-smi` may be absent, may not support a field on a given driver, and
returns "[N/A]" for several fields inside a MIG instance. None of that is worth
failing a benchmark over, so every value is Optional and every failure is
recorded in the `error` key rather than raised. A missing clock reading must
never be the reason a two-hour sweep dies.
"""
from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time

# nvidia-smi field -> the column name we publish it under. Ordered, because the
# query string and the parsed result have to line up positionally.
_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "gpu_name"),
    ("driver_version", "driver_version"),
    ("persistence_mode", "persistence_mode"),
    ("clocks.applications.graphics", "clock_applications_mhz"),
    ("clocks.max.graphics", "clock_max_graphics_mhz"),
    ("clocks.max.sm", "clock_max_sm_mhz"),
    ("clocks.current.graphics", "clock_current_graphics_mhz"),
    ("clocks.current.sm", "clock_current_sm_mhz"),
    ("clocks.current.memory", "clock_current_memory_mhz"),
    ("power.limit", "power_limit_w"),
    ("power.draw", "power_draw_w"),
    ("temperature.gpu", "temperature_c"),
    ("pstate", "pstate"),
    ("clocks_throttle_reasons.active", "throttle_reasons_hex"),
)

_NUMERIC = {
    "clock_applications_mhz", "clock_max_graphics_mhz", "clock_max_sm_mhz",
    "clock_current_graphics_mhz", "clock_current_sm_mhz",
    "clock_current_memory_mhz", "power_limit_w", "power_draw_w",
    "temperature_c",
}

_NA = {"[n/a]", "n/a", "[not supported]", "not supported", "unknown", ""}


def _smi(query: str, timeout: float = 5.0) -> str | None:
    """One nvidia-smi query, or None if it is unavailable for any reason."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout


def _coerce(col: str, raw: str):
    v = raw.strip()
    if v.lower() in _NA:
        return None
    if col in _NUMERIC:
        try:
            return float(v)
        except ValueError:
            return None
    return v


def probe(index: int = 0) -> dict:
    """Static GPU configuration for one physical GPU, as publishable columns.

    `index` is a PHYSICAL GPU index, not a MIG instance: nvidia-smi reports clocks
    and persistence mode per card, and a MIG slice inherits the card's. On a host
    where CUDA_VISIBLE_DEVICES hides the card this may not be the one being timed;
    `gpu_name` is recorded so that mismatch is visible rather than assumed away.
    """
    state: dict = {col: None for _, col in _FIELDS}
    state["gpustate_error"] = None

    raw = _smi(",".join(field for field, _ in _FIELDS))
    if raw is None:
        state["gpustate_error"] = "nvidia-smi unavailable"
        return state

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if index >= len(lines):
        state["gpustate_error"] = f"gpu index {index} not in nvidia-smi output ({len(lines)} gpus)"
        return state

    parts = lines[index].split(",")
    if len(parts) != len(_FIELDS):
        state["gpustate_error"] = f"expected {len(_FIELDS)} fields, got {len(parts)}"
        return state

    for (_, col), raw_val in zip(_FIELDS, parts):
        state[col] = _coerce(col, raw_val)
    return state


def clock_headroom(state: dict) -> float | None:
    """Applications clock as a fraction of the card's max. None if unknown.

    This is the single number that separates "this GPU was allowed to run" from
    "this GPU was pinned low and the benchmark measured the pin". 765/1410 = 0.54
    is the value that made the private server look 3.3x slower than a MIG slice.
    """
    app, mx = state.get("clock_applications_mhz"), state.get("clock_max_graphics_mhz")
    if not app or not mx:
        return None
    return round(app / mx, 4)


class ClockSampler:
    """Poll the SM clock on a background thread while something else is timed.

    THE POINT: an applications clock read at idle says what the GPU is ALLOWED to
    do. It does not say what it did. A bursty batch-1 workload can leave a card
    parked at its idle clock for the whole run -- which is exactly the hypothesis
    on record for the private server, and the reason its numbers were never
    trusted. Sampling under load is what turns that from a suspicion into a
    measurement.

    Deliberately coarse: one nvidia-smi call per interval on a daemon thread.
    That is ~1-2 ms of subprocess work every 200 ms, on a different thread from
    the CUDA calls, and it does not enter the timed region -- the timing loop
    itself is bracketed by cuda.synchronize() and unaffected by a sampler that
    never touches the CUDA context.

        with ClockSampler() as cs:
            result = benchmark_speed(...)
        result.update(cs.summary())
    """

    def __init__(self, index: int = 0, interval: float = 0.2):
        self.index = index
        self.interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            raw = _smi("clocks.current.sm", timeout=2.0)
            if raw is not None:
                lines = [ln for ln in raw.splitlines() if ln.strip()]
                if self.index < len(lines):
                    v = _coerce("clock_current_sm_mhz", lines[self.index])
                    if v is not None:
                        self._samples.append(v)
            # wait() rather than sleep(): stop() takes effect immediately instead
            # of after one more full interval.
            self._stop.wait(self.interval)

    def start(self) -> "ClockSampler":
        if shutil.which("nvidia-smi") is None:
            return self
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "ClockSampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return self

    def __enter__(self) -> "ClockSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def summary(self) -> dict:
        """Columns describing the clock DURING the run. All None if unsampled."""
        if not self._samples:
            return {"sm_clock_under_load_median_mhz": None,
                    "sm_clock_under_load_max_mhz": None,
                    "sm_clock_under_load_min_mhz": None,
                    "sm_clock_samples": 0}
        return {
            "sm_clock_under_load_median_mhz": float(statistics.median(self._samples)),
            "sm_clock_under_load_max_mhz": float(max(self._samples)),
            "sm_clock_under_load_min_mhz": float(min(self._samples)),
            "sm_clock_samples": len(self._samples),
        }


def describe(state: dict) -> str:
    """One human-readable line for the benchmark's stdout."""
    if state.get("gpustate_error"):
        return f"  GPU state unavailable: {state['gpustate_error']}"
    hr = clock_headroom(state)
    bits = [
        f"persistence={state.get('persistence_mode')}",
        f"app_clock={state.get('clock_applications_mhz')}/{state.get('clock_max_graphics_mhz')} MHz"
        + (f" ({hr:.0%})" if hr is not None else ""),
        f"power_limit={state.get('power_limit_w')} W",
    ]
    warn = ""
    if hr is not None and hr < 0.95:
        warn = ("\n  WARNING  applications clock is pinned below the card's max. A batch-1 "
                "benchmark never sustains demand, so this number measures the pin, not the "
                "model. Compare only against rows with the same clock state.")
    if str(state.get("persistence_mode", "")).lower() == "disabled":
        warn += ("\n  WARNING  persistence_mode is Disabled. The driver unloads between "
                 "processes and the clock ramps late or not at all.")
    return "  GPU state: " + "  ".join(bits) + warn


def sampler_warning(summary: dict, state: dict) -> str | None:
    """Compare what the clock DID against what it was allowed to do."""
    med = summary.get("sm_clock_under_load_median_mhz")
    mx = state.get("clock_max_sm_mhz") or state.get("clock_max_graphics_mhz")
    if not med or not mx:
        return None
    if med / mx < 0.9:
        return (f"  UNDER LOAD  median SM clock {med:.0f} MHz against a {mx:.0f} MHz max "
                f"({med / mx:.0%}). The GPU did not ramp during the benchmark -- this is a "
                f"clock result, not an architecture result.")
    return None
