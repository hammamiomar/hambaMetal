"""
torch mps profiler for performance analysis
tracks timing and memory usage across pipeline components
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

import torch

from ..config import ProfilingConfig


@dataclass
class TimingRecord:
    """single timing measurement"""

    name: str
    duration_ms: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """memory usage at a point in time"""

    allocated_mb: float
    reserved_mb: float
    timestamp: float


class MPSProfiler:
    """
    performance profiler for mps (apple silicon)
    tracks per-module timing and memory usage

    usage:
        profiler = MPSProfiler()
        with profiler.profile("unet_forward"):
            output = unet(input)
        profiler.print_summary()
    """

    def __init__(self, config: Optional[ProfilingConfig] = None):
        self.config = config or ProfilingConfig()

        # timing records
        self.timings: List[TimingRecord] = []
        self.timing_sums: Dict[str, float] = {}
        self.timing_counts: Dict[str, int] = {}

        # memory tracking
        self.memory_snapshots: List[MemorySnapshot] = []

        # device detection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.enabled = config.enable_profiling if config else True

    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict] = None):
        """
        context manager for timing code blocks

        usage:
            with profiler.profile("vae_decode"):
                image = vae.decode(latent)
        """
        if not self.enabled:
            yield
            return

        # start timing
        if self.device == "mps":
            start_event = torch.mps.Event(enable_timing=True)
            end_event = torch.mps.Event(enable_timing=True)
            start_event.record()
        elif self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            # cpu fallback
            start_time = time.perf_counter()

        # execute block
        try:
            yield
        finally:
            # end timing
            if self.device == "mps":
                end_event.record()
                torch.mps.synchronize()
                duration_ms = start_event.elapsed_time(end_event)
            elif self.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                duration_ms = start_event.elapsed_time(end_event)
            else:
                duration_ms = (time.perf_counter() - start_time) * 1000

            # record
            self._add_timing(name, duration_ms, metadata or {})

            # memory snapshot if enabled
            if self.config.track_memory:
                self._take_memory_snapshot()

    def _add_timing(self, name: str, duration_ms: float, metadata: Dict) -> None:
        """add timing record"""
        record = TimingRecord(
            name=name,
            duration_ms=duration_ms,
            timestamp=time.time(),
            metadata=metadata,
        )

        self.timings.append(record)

        # update running stats
        if name not in self.timing_sums:
            self.timing_sums[name] = 0.0
            self.timing_counts[name] = 0

        self.timing_sums[name] += duration_ms
        self.timing_counts[name] += 1

    def _take_memory_snapshot(self) -> None:
        """capture current memory usage"""
        if self.device == "mps":
            allocated = torch.mps.current_allocated_memory() / (1024**2)  # mb
            reserved = torch.mps.driver_allocated_memory() / (1024**2)
        elif self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
        else:
            # cpu - no tracking
            return

        snapshot = MemorySnapshot(
            allocated_mb=allocated,
            reserved_mb=reserved,
            timestamp=time.time(),
        )

        self.memory_snapshots.append(snapshot)

    def get_avg_time(self, name: str) -> float:
        """get average time for a specific operation"""
        if name not in self.timing_counts or self.timing_counts[name] == 0:
            return 0.0

        return self.timing_sums[name] / self.timing_counts[name]

    def get_total_time(self, name: str) -> float:
        """get total time for a specific operation"""
        return self.timing_sums.get(name, 0.0)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        get summary statistics for all profiled operations

        returns:
            dict mapping operation name to {avg_ms, total_ms, count}
        """
        summary = {}

        for name in self.timing_sums.keys():
            count = self.timing_counts[name]
            total = self.timing_sums[name]
            avg = total / count if count > 0 else 0.0

            summary[name] = {
                "avg_ms": avg,
                "total_ms": total,
                "count": count,
                "total_seconds": total / 1000.0,
            }

        return summary

    def print_summary(self) -> None:
        """print formatted summary to console"""
        summary = self.get_summary()

        if not summary:
            print("no profiling data collected")
            return

        print("\n" + "=" * 70)
        print("performance profiling summary")
        print("=" * 70)

        # sort by total time descending
        sorted_ops = sorted(
            summary.items(),
            key=lambda x: x[1]["total_ms"],
            reverse=True,
        )

        print(f"{'operation':<30} {'avg (ms)':>12} {'total (s)':>12} {'count':>8}")
        print("-" * 70)

        for name, stats in sorted_ops:
            print(
                f"{name:<30} {stats['avg_ms']:>12.2f} {stats['total_seconds']:>12.2f} {stats['count']:>8}"
            )

        # total time
        total_time_s = sum(s["total_seconds"] for s in summary.values())
        print("-" * 70)
        print(f"{'TOTAL':<30} {'':<12} {total_time_s:>12.2f}")

        # memory summary if available
        if self.memory_snapshots:
            print("\n" + "=" * 70)
            print("memory usage")
            print("=" * 70)

            peak_allocated = max(s.allocated_mb for s in self.memory_snapshots)
            peak_reserved = max(s.reserved_mb for s in self.memory_snapshots)

            print(f"peak allocated: {peak_allocated:.2f} mb")
            print(f"peak reserved:  {peak_reserved:.2f} mb")

        print("=" * 70 + "\n")

    def export_json(self, path: Optional[str] = None) -> None:
        """export profiling data to json"""
        if not self.enabled:
            return

        path = path or self.config.json_path
        output = {
            "summary": self.get_summary(),
            "device": self.device,
            "num_timings": len(self.timings),
            "num_memory_snapshots": len(self.memory_snapshots),
        }

        # add detailed timings if requested
        if len(self.timings) < 10000:  # avoid huge files
            output["detailed_timings"] = [asdict(t) for t in self.timings]

        # add memory snapshots
        if self.memory_snapshots:
            output["memory_snapshots"] = [asdict(s) for s in self.memory_snapshots]

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"profiling data exported to {path}")

    def reset(self) -> None:
        """reset all profiling data"""
        self.timings.clear()
        self.timing_sums.clear()
        self.timing_counts.clear()
        self.memory_snapshots.clear()


# global profiler instance
_global_profiler: Optional[MPSProfiler] = None


def get_profiler() -> MPSProfiler:
    """get or create global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MPSProfiler()
    return _global_profiler


def reset_profiler() -> None:
    """reset global profiler"""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.reset()
