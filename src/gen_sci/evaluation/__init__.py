"""Benchmark evaluation and reporting."""

from gen_sci.evaluation.generative_metrics import (
    CoverageDensityResult,
    coverage_and_density,
    mmd,
    wasserstein_1d,
)
from gen_sci.evaluation.report import write_report
from gen_sci.evaluation.runner import run_benchmark

__all__ = [
    "CoverageDensityResult",
    "coverage_and_density",
    "mmd",
    "run_benchmark",
    "wasserstein_1d",
    "write_report",
]
