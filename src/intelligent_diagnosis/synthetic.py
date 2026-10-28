"""Original simulated device events, not production traces or fault-code lookups.

Signals reflect deliberately simplified electrical scenarios. Class identity is
never a feature; noisy DTC activity is shared across faults and normal events.
These data support software demonstrations, not field-accuracy claims.
"""

from hashlib import sha256

import numpy as np

from .schemas import DiagnosisRequest, Domain, Entry, SignalSample

SYNTHETIC_VERSION = "electrical-simulation-v1"
CLASSES: dict[Domain, tuple[str, ...]] = {
    Domain.SAFETY: ("circuit_open", "circuit_short", "safety_sensor_invalid", "safety_normal"),
    Domain.LOW_VOLTAGE: (
        "battery_depleted", "charging_failure", "terminal_resistance", "voltage_normal"
    ),
}
CLASS_DOMAIN = {class_id: domain for domain, classes in CLASSES.items() for class_id in classes}
# Domain-level identifiers, not class labels. These match the initial original
# knowledge snapshot; callers using another snapshot must bind its fault IDs.
FAULT_IDS = {
    Domain.SAFETY: "fault-safety-demo",
    Domain.LOW_VOLTAGE: "fault-voltage-demo",
}
COMPLAINTS: dict[str, tuple[str, ...]] = {
    "circuit_open": (
        "设备安全回路报警，连锁没有响应", "安全联锁指示灯亮着，回路检测异常",
        "保护电路出现警告，请检查安全回路",
    ),
    "circuit_short": (
        "安全联锁回路报警，保护功能异常", "安全回路显示故障，连锁动作不正常",
        "安全保护线路告警，需要排查回路",
    ),
    "safety_sensor_invalid": (
        "安全传感器的读数跳动，出现保护报警", "联锁传感器供电异常，安全灯亮了",
        "安全检测信号不稳定，回路状态异常",
    ),
    "safety_normal": (
        "请诊断安全回路的当前状态", "检查联锁保护是否工作正常",
        "我想确认安全传感器现在有无故障",
    ),
    "battery_depleted": (
        "蓄电池电压低，设备启动没力气", "低压电源报警，电池电量不足",
        "设备启动困难，请检查蓄电池电压",
    ),
    "charging_failure": (
        "运行时充电指示异常，电瓶电压上不去", "发动机运行但电池没有充进电",
        "充电系统报警，请诊断低压供电",
    ),
    "terminal_resistance": (
        "蓄电池接线柱发热，供电忽高忽低", "电瓶端子接触异常，低压供电不稳",
        "电池端子压降变大，设备启动困难",
    ),
    "voltage_normal": (
        "请检查当前蓄电池供电是否正常", "诊断低压系统现在的电压状态",
        "确认充电和电池端子有没有故障",
    ),
}


def _device_rng(device_id: str) -> np.random.Generator:
    seed = int.from_bytes(sha256(device_id.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def make_request(
    class_id: str,
    seed: int = 0,
    request_id: str | None = None,
    device_id: str | None = None,
    entry: Entry = Entry.USER,
) -> DiagnosisRequest:
    """Generate nine samples spanning exactly ±1000ms around a synthetic event.

    A device has a stable calibration offset across events, independent of the
    class. Seed controls event noise, loads and complaint variants. Identifiers
    contain no class label; they and complaint text are excluded from the models.
    """
    if class_id not in CLASS_DOMAIN:
        raise ValueError(f"unknown synthetic class: {class_id}")
    domain = CLASS_DOMAIN[class_id]
    rng = np.random.default_rng(seed)
    device_id = device_id or f"sim-device-{seed:08d}"
    request_id = request_id or f"sim-event-{seed:08d}"
    device_rng = _device_rng(device_id)
    voltage_offset = float(device_rng.normal(0, 0.10))
    resistance_scale = float(device_rng.uniform(0.9, 1.1))
    current_offset = float(device_rng.normal(0, 0.65))
    fault_time = 10_000 + int(rng.integers(0, 100_000))
    offsets = np.linspace(-1000, 1000, 9, dtype=int)
    event_phase = rng.uniform(-1, 1)
    fault = class_id not in {"safety_normal", "voltage_normal"}
    # Generic diagnostic activity is fallible and non-specific: codes 1..4 are
    # sampled independently of fault identity, including spurious normal codes.
    dtc_probability = 0.65 if fault else 0.18
    event_warning = fault and bool(rng.random() < 0.94)
    samples = []
    if domain == Domain.SAFETY:
        if class_id == "circuit_open":
            resistance = rng.uniform(1800, 5200) * resistance_scale
            resistance_noise = 0.035 * resistance
            supply = rng.uniform(11.6, 13.9) + voltage_offset
        elif class_id == "circuit_short":
            resistance = rng.uniform(0.6, 5.5) * resistance_scale
            resistance_noise = 0.25
            supply = rng.uniform(11.6, 13.9) + voltage_offset
        elif class_id == "safety_sensor_invalid":
            resistance = rng.uniform(50, 190) * resistance_scale
            resistance_noise = rng.uniform(20, 55)
            supply = rng.uniform(3.2, 7.3) + voltage_offset
        else:
            resistance = rng.uniform(88, 130) * resistance_scale
            resistance_noise = rng.uniform(0.4, 1.6)
            supply = rng.uniform(11.8, 13.7) + voltage_offset
        for offset in offsets:
            values = {
                "resistance_ohm": max(0.05, float(rng.normal(resistance, resistance_noise))),
                "supply_voltage": float(rng.normal(supply, 0.09)),
                "warning_active": float(event_warning),
                "dtc": float(rng.integers(1, 5)) if rng.random() < dtc_probability else 0.0,
            }
            samples.append(SignalSample(timestamp_ms=fault_time + int(offset), values=values))
    else:
        if class_id == "battery_depleted":
            voltage = rng.uniform(9.1, 11.25) + voltage_offset
            current = rng.uniform(-24, -4) + current_offset
            running = 0.0
            drop = rng.uniform(0.02, 0.14)
            slope = rng.uniform(-0.24, -0.03)
        elif class_id == "charging_failure":
            voltage = rng.uniform(11.55, 12.5) + voltage_offset
            current = rng.uniform(-18, -3) + current_offset
            running = 1.0
            drop = rng.uniform(0.02, 0.14)
            slope = rng.uniform(-0.16, -0.01)
        elif class_id == "terminal_resistance":
            voltage = rng.uniform(10.8, 12.4) + voltage_offset
            current = rng.uniform(-25, -5) + current_offset
            running = float(rng.integers(0, 2))
            drop = rng.uniform(0.75, 1.7)
            slope = rng.uniform(-0.22, 0.22)
        else:
            running = float(rng.integers(0, 2))
            voltage = rng.uniform(13.65, 14.45) if running else rng.uniform(12.65, 13.0)
            voltage += voltage_offset
            current = rng.uniform(3, 14) if running else rng.uniform(-2, 0)
            current += current_offset
            drop = rng.uniform(0.02, 0.14)
            slope = rng.uniform(-0.025, 0.025)
        for offset in offsets:
            wave = np.sin(offset / 1000 * np.pi + event_phase)
            values = {
                "voltage": float(voltage + slope * offset / 1000 + rng.normal(0, 0.035)),
                "current_a": float(current + wave * 0.6 + rng.normal(0, 0.15)),
                "engine_running": running,
                "terminal_drop_v": max(0.001, float(drop + rng.normal(0, 0.012))),
                "warning_active": float(event_warning),
                "dtc": float(rng.integers(1, 5)) if rng.random() < dtc_probability else 0.0,
            }
            samples.append(SignalSample(timestamp_ms=fault_time + int(offset), values=values))
    return DiagnosisRequest(
        request_id=request_id,
        device_id=device_id,
        entry=entry,
        complaint=COMPLAINTS[class_id][int(rng.integers(len(COMPLAINTS[class_id])))],
        domain=domain if entry == Entry.TELEMETRY else None,
        fault_id=FAULT_IDS[domain] if entry == Entry.TELEMETRY else None,
        fault_time_ms=fault_time,
        samples=samples,
    )
