"""合成门店的只读能力匹配；候选排名不代表预约、维修或库存占用。"""

from copy import deepcopy
from math import asin, cos, isfinite, radians, sin, sqrt
from numbers import Real

from .schemas import Attribution, DomainError

# 坐标、库存及店名均为原创演示数据，不声称对应真实经营门店。
DEFAULT_SERVICES = [
    {
        "service_id": "demo-riverside",
        "name": "澄川合成服务站",
        "skills": ["safety_loop", "low_voltage"],
        "parts": {
            "loop_harness": 3,
            "safety_sensor": 2,
            "battery": 2,
            "charging_unit": 1,
            "terminal_kit": 4,
        },
        "capacity": 4,
        "active_jobs": 1,
        "latitude": 31.02,
        "longitude": 121.01,
    },
    {
        "service_id": "demo-meadow",
        "name": "青禾合成检修站",
        "skills": ["low_voltage"],
        "parts": {"battery": 4, "charging_unit": 2, "terminal_kit": 6},
        "capacity": 3,
        "active_jobs": 0,
        "latitude": 30.98,
        "longitude": 121.06,
    },
    {
        "service_id": "demo-north",
        "name": "北岚合成安全站",
        "skills": ["safety_loop"],
        "parts": {"loop_harness": 2, "safety_sensor": 3},
        "capacity": 2,
        "active_jobs": 1,
        "latitude": 31.08,
        "longitude": 120.97,
    },
    {
        "service_id": "demo-lake",
        "name": "湖汀合成服务站",
        "skills": ["safety_loop", "low_voltage"],
        "parts": {
            "loop_harness": 1,
            "safety_sensor": 0,
            "battery": 1,
            "charging_unit": 0,
            "terminal_kit": 2,
        },
        "capacity": 5,
        "active_jobs": 4,
        "latitude": 30.94,
        "longitude": 120.95,
    },
]


def _coordinate(value: object, bound: float) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and isfinite(value)
        and -bound <= value <= bound
    )


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """平均地球半径6371.0088km的haversine球面距离，非道路里程。"""
    delta_lat, delta_lon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(delta_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(max(0.0, min(1.0, a))))


def rank_services(
    attribution: Attribution,
    latitude: float,
    longitude: float,
    services: list[dict] | None = None,
) -> list[dict]:
    """先硬过滤技能/库存/余量，再综合球面距离、负载和库存评分。

    score = 55/(1+距离km/20) + 30*(1-负载率) + 15*库存充足度。
    所需配件三件及以上视为库存充足，无配件需求时该项为1。
    services=[]明确表示没有门店；只有None才启用合成默认目录。
    """
    if not isinstance(attribution, Attribution):
        raise DomainError("invalid_attribution", "派单候选必须绑定业务归因", 422)
    if not _coordinate(latitude, 90) or not _coordinate(longitude, 180):
        raise DomainError("invalid_coordinates", "经纬度须为范围内的有限数值", 422)
    if services is not None and not isinstance(services, list):
        raise DomainError("invalid_services", "门店目录必须为列表", 422)

    candidates: list[dict] = []
    seen: set[str] = set()
    for service in DEFAULT_SERVICES if services is None else services:
        # 目录缺项或不可信时不猜测能力/位置，不将其列为候选。
        if not isinstance(service, dict):
            continue
        service_id = service.get("service_id")
        if not isinstance(service_id, str) or not service_id.strip():
            continue
        if service_id in seen:
            raise DomainError("duplicate_service", "门店目录存在重复服务标识", 422)
        seen.add(service_id)
        skills, parts = service.get("skills"), service.get("parts")
        capacity, active = service.get("capacity"), service.get("active_jobs")
        if (
            not isinstance(service.get("name"), str)
            or not service["name"].strip()
            or not isinstance(skills, list)
            or any(not isinstance(skill, str) for skill in skills)
            or attribution.required_skill not in skills
            or not isinstance(parts, dict)
            or not _count(capacity)
            or capacity == 0
            or not _count(active)
            or active >= capacity
            or not _coordinate(service.get("latitude"), 90)
            or not _coordinate(service.get("longitude"), 180)
        ):
            continue
        part = attribution.required_part
        stock = parts.get(part) if part else None
        if part and (not _count(stock) or stock < 1):
            continue
        distance = _distance_km(latitude, longitude, service["latitude"], service["longitude"])
        workload = active / capacity
        inventory = min(stock / 3, 1.0) if part else 1.0
        score = 55 / (1 + distance / 20) + 30 * (1 - workload) + 15 * inventory
        stock_note = f"所需配件库存{stock}件" if part else "该归因无配件库存要求"
        candidate = deepcopy(service)
        candidate.update(
            {
                "distance_km": round(distance, 6),
                "workload_ratio": round(workload, 6),
                "available_capacity": capacity - active,
                "score": round(score, 6),
                "explanation": (
                    f"具备所需技能；球面距离{distance:.2f}公里（非道路里程）；"
                    f"工位负载{active}/{capacity}；{stock_note}。"
                    f"距离、空闲比例、库存按55/30/15加权，得分{score:.2f}。"
                    "仅为候选，尚未预约；库存与工位须在服务事务中重新校验并占用。"
                ),
            }
        )
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: (-item["score"], item["service_id"]))
