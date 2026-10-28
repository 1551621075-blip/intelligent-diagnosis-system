"""原创固定窗口统计特征。缺测不补零，质量问题由调用方在模型之前拒识。"""

from collections import defaultdict
from math import isfinite, nan

import numpy as np

from .schemas import DiagnosisRequest, Domain

FEATURE_SCHEMA_VERSION = "window-v1"
WINDOW_HALF_WIDTH_MS = 1000
MIN_UNIQUE_POINTS = 3
MAX_EDGE_GAP_MS = 500

REQUIRED_SIGNALS: dict[Domain, tuple[str, ...]] = {
    Domain.SAFETY: ("resistance_ohm", "supply_voltage", "warning_active", "dtc"),
    Domain.LOW_VOLTAGE: (
        "voltage", "current_a", "engine_running", "terminal_drop_v", "warning_active", "dtc"
    ),
}
FEATURE_COLUMNS: dict[Domain, tuple[str, ...]] = {
    Domain.SAFETY: (
        "resistance_ohm_median", "resistance_ohm_min", "resistance_ohm_max",
        "resistance_ohm_std", "supply_voltage_min", "supply_voltage_mean",
        "warning_active_max", "dtc_active_fraction",
    ),
    Domain.LOW_VOLTAGE: (
        "voltage_min", "voltage_mean", "voltage_std", "voltage_slope_v_per_s",
        "current_a_mean", "current_a_min", "engine_running_max",
        "terminal_drop_v_max", "terminal_drop_v_mean", "warning_active_max",
        "dtc_active_fraction",
    ),
}


def extract_features(
    request: DiagnosisRequest, domain: Domain
) -> tuple[dict[str, float], list[str]]:
    """Only use [fault_time - 1000, fault_time + 1000], inclusive.

    Sort and deduplicate timestamps, reporting duplicates (including conflicting
    readings) rather than letting retransmission change the statistics. Three
    unique points and observations on both sides, within 500ms of each window
    edge, are required. Missing/nonfinite values invalidate that signal's entire
    feature family; they remain NaN, not plausible-looking imputed measurements.
    """
    domain = Domain(domain)
    issues: list[str] = []
    if request.domain is not None and request.domain != domain:
        issues.append("domain_mismatch")
    start = request.fault_time_ms - WINDOW_HALF_WIDTH_MS
    end = request.fault_time_ms + WINDOW_HALF_WIDTH_MS
    grouped = defaultdict(list)
    for sample in request.samples:
        if start <= sample.timestamp_ms <= end:
            grouped[sample.timestamp_ms].append(sample)
    points = []
    conflicts: set[str] = set()
    for timestamp in sorted(grouped):
        rows = grouped[timestamp]
        points.append(rows[0])
        if len(rows) > 1:
            issues.append(f"duplicate_timestamp:{timestamp}")
            for signal in REQUIRED_SIGNALS[domain]:
                if any(row.values.get(signal) != rows[0].values.get(signal) for row in rows[1:]):
                    conflicts.add(signal)
                    issues.append(f"conflicting_timestamp:{timestamp}:{signal}")
    if len(points) < MIN_UNIQUE_POINTS:
        issues.append("insufficient_unique_points")
    if (
        not points
        or points[0].timestamp_ms > start + MAX_EDGE_GAP_MS
        or points[-1].timestamp_ms < end - MAX_EDGE_GAP_MS
        or points[0].timestamp_ms >= request.fault_time_ms
        or points[-1].timestamp_ms <= request.fault_time_ms
    ):
        issues.append("insufficient_time_coverage")
    signals: dict[str, np.ndarray] = {}
    for signal in REQUIRED_SIGNALS[domain]:
        values = []
        invalid = signal in conflicts
        for sample in points:
            value = sample.values.get(signal)
            if value is None:
                issues.append(f"missing_signal:{signal}:{sample.timestamp_ms}")
                invalid = True
            elif not isfinite(value):
                issues.append(f"nonfinite_signal:{signal}:{sample.timestamp_ms}")
                invalid = True
            elif (signal in {"warning_active", "engine_running"} and value not in {0, 1}) or (
                signal == "dtc" and (value < 0 or not float(value).is_integer())
            ):
                issues.append(f"invalid_signal:{signal}:{sample.timestamp_ms}")
                invalid = True
            else:
                values.append(value)
        if not points:
            issues.append(f"missing_signal:{signal}")
        signals[signal] = np.asarray([] if invalid else values, dtype=float)

    features = dict.fromkeys(FEATURE_COLUMNS[domain], nan)

    def aggregate(signal: str, operation: str) -> float:
        values = signals[signal]
        if not len(values):
            return nan
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(getattr(np, operation)(values))
        return value

    if domain == Domain.SAFETY:
        for operation in ("median", "min", "max", "std"):
            features[f"resistance_ohm_{operation}"] = aggregate("resistance_ohm", operation)
        for operation in ("min", "mean"):
            features[f"supply_voltage_{operation}"] = aggregate("supply_voltage", operation)
    else:
        for operation in ("min", "mean", "std"):
            features[f"voltage_{operation}"] = aggregate("voltage", operation)
        for operation in ("mean", "min"):
            features[f"current_a_{operation}"] = aggregate("current_a", operation)
        features["engine_running_max"] = aggregate("engine_running", "max")
        for operation in ("max", "mean"):
            features[f"terminal_drop_v_{operation}"] = aggregate("terminal_drop_v", operation)
        if len(signals["voltage"]) >= 2:
            times = np.asarray([point.timestamp_ms - request.fault_time_ms for point in points])
            times = times.astype(float) / 1000
            centered = times - times.mean()
            with np.errstate(over="ignore", invalid="ignore"):
                features["voltage_slope_v_per_s"] = float(
                    np.dot(centered, signals["voltage"]) / np.dot(centered, centered)
                )
    features["warning_active_max"] = aggregate("warning_active", "max")
    if len(signals["dtc"]):
        features["dtc_active_fraction"] = float(np.mean(signals["dtc"] > 0))
    for key, value in features.items():
        if not isfinite(value):
            features[key] = nan
            issues.append(f"unavailable_feature:{key}")
    return features, list(dict.fromkeys(issues))
