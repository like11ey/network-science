"""metrics 包：跨 phase 的度量工具（韧性指标、相变阈值、恢复过程）。

放在 src/ 顶层而不是某个 phaseN 里，是因为 L0–L3 四级台阶、phase4 稳健性、
phase5 级联都要用同一套口径 —— 口径只能有一份。
"""

from .resilience_metrics import (  # noqa: F401
    jump_size, max_slope, transition_order,
    threshold_by_midpoint, threshold_by_max_drop, percolation_threshold,
    loss_area, rhi, recovery_slope, recovery_time, recovery_metrics,
    vulnerability_probe,
)

__all__ = [
    "jump_size", "max_slope", "transition_order",
    "threshold_by_midpoint", "threshold_by_max_drop", "percolation_threshold",
    "loss_area", "rhi", "recovery_slope", "recovery_time", "recovery_metrics",
    "vulnerability_probe",
]
