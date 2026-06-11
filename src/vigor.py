from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class VigorMetrics:
    final_germination_rate: float
    t50_h: float | None
    mean_germination_time_h: float
    germination_speed_index: float
    uniformity_score: float


def _interpolate_t50(summary: pd.DataFrame) -> float | None:
    if summary.empty or summary["germination_rate"].max() < 50.0:
        return None

    ordered = summary.sort_values("time_h")
    previous_time = float(ordered.iloc[0]["time_h"])
    previous_rate = float(ordered.iloc[0]["germination_rate"])

    if previous_rate >= 50.0:
        return previous_time

    for _, row in ordered.iloc[1:].iterrows():
        current_time = float(row["time_h"])
        current_rate = float(row["germination_rate"])
        if current_rate >= 50.0:
            if current_rate == previous_rate:
                return current_time
            ratio = (50.0 - previous_rate) / (current_rate - previous_rate)
            return previous_time + ratio * (current_time - previous_time)
        previous_time = current_time
        previous_rate = current_rate
    return None


def compute_vigor_metrics(summary: pd.DataFrame) -> VigorMetrics:
    if summary.empty:
        return VigorMetrics(0.0, None, 0.0, 0.0, 0.0)

    ordered = summary.sort_values("time_h").reset_index(drop=True)
    final_rate = float(ordered.iloc[-1]["germination_rate"])
    t50 = _interpolate_t50(ordered)

    cumulative = ordered["germinated"].astype(float)
    newly_germinated = cumulative.diff().fillna(cumulative.iloc[0]).clip(lower=0)
    times = ordered["time_h"].astype(float).replace(0, 0.5)

    total_new = float(newly_germinated.sum())
    if total_new > 0:
        mean_time = float((newly_germinated * ordered["time_h"].astype(float)).sum() / total_new)
    else:
        mean_time = 0.0

    speed_index = float((newly_germinated / times).sum())

    active_times = ordered.loc[newly_germinated > 0, "time_h"].astype(float)
    if len(active_times) <= 1:
        uniformity = 100.0 if len(active_times) == 1 else 0.0
    else:
        span = max(1.0, float(active_times.max() - active_times.min()))
        uniformity = max(0.0, 100.0 - span * 3.0)

    return VigorMetrics(final_rate, t50, mean_time, speed_index, uniformity)


def score_vigor(metrics: VigorMetrics) -> float:
    final_score = metrics.final_germination_rate

    if metrics.t50_h is None:
        speed_score = 0.0
    else:
        speed_score = max(0.0, min(100.0, 100.0 - metrics.t50_h * 2.0))

    uniformity_score = max(0.0, min(100.0, metrics.uniformity_score))
    score = 0.5 * final_score + 0.3 * speed_score + 0.2 * uniformity_score
    return max(0.0, min(100.0, score))
