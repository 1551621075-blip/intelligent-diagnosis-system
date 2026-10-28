from copy import deepcopy

import pytest

from intelligent_diagnosis.dispatch import rank_services
from intelligent_diagnosis.knowledge import KnowledgeBase
from intelligent_diagnosis.schemas import DomainError


@pytest.fixture
def attribution():
    return KnowledgeBase().resolve("low_voltage", "battery_depleted")


def service(identifier="demo-test", **changes):
    result = {
        "service_id": identifier,
        "name": "合成测试站",
        "skills": ["low_voltage"],
        "parts": {"battery": 3},
        "capacity": 4,
        "active_jobs": 0,
        "latitude": 0.0,
        "longitude": 0.0,
    }
    result.update(changes)
    return result


def test_default_candidates_and_explicit_empty_directory(attribution):
    result = rank_services(attribution, 31, 121)
    assert 1 <= len(result) <= 5
    assert all("合成" in item["name"] for item in result)
    assert all(item["parts"]["battery"] > 0 for item in result)
    assert rank_services(attribution, 31, 121, []) == []


@pytest.mark.parametrize(
    "change",
    [
        {"skills": ["safety_loop"]},
        {"skills": "low_voltage"},
        {"parts": {}},
        {"parts": {"battery": 0}},
        {"parts": {"battery": -1}},
        {"parts": {"battery": True}},
        {"capacity": 0},
        {"active_jobs": 4},
        {"active_jobs": 8},
        {"active_jobs": -1},
        {"capacity": "4"},
        {"latitude": 91},
        {"longitude": float("nan")},
        {"name": ""},
    ],
)
def test_missing_capability_is_filtered_not_fabricated(attribution, change):
    assert rank_services(attribution, 0, 0, [service(**change)]) == []


def test_haversine_distance_and_score(attribution):
    result = rank_services(
        attribution, 0, 0, [service("origin"), service("one-degree", longitude=1)]
    )
    assert [item["service_id"] for item in result] == ["origin", "one-degree"]
    assert result[0]["distance_km"] == 0
    assert result[0]["score"] == 100
    assert result[1]["distance_km"] == pytest.approx(111.19508, abs=1e-5)
    expected = 55 / (1 + 111.19508 / 20) + 30 + 15
    assert result[1]["score"] == pytest.approx(expected, abs=1e-5)


def test_antimeridian_distance_is_short_path(attribution):
    result = rank_services(attribution, 0, 179.5, [service(longitude=-179.5)])
    assert result[0]["distance_km"] == pytest.approx(111.19508, abs=1e-5)


def test_workload_and_inventory_affect_ranking(attribution):
    services = [
        service("busy", active_jobs=3),
        service("sparse", parts={"battery": 1}),
        service("ready"),
    ]
    result = rank_services(attribution, 0, 0, services)
    assert [item["service_id"] for item in result] == ["ready", "sparse", "busy"]
    assert result[0]["available_capacity"] == 4
    assert result[-1]["workload_ratio"] == 0.75


def test_deterministic_ties_no_reservation_and_no_mutation(attribution):
    services = [service("b"), service("a")]
    original = deepcopy(services)
    result = rank_services(attribution, 0, 0, services)
    assert [item["service_id"] for item in result] == ["a", "b"]
    assert services == original
    assert all("尚未预约" in item["explanation"] for item in result)
    assert not any("reservation_id" in item for item in result)
    result[0]["parts"]["battery"] = 0
    assert services == original


def test_no_part_requirement_does_not_fabricate_stock():
    normal = KnowledgeBase().resolve("low_voltage", "voltage_normal")
    result = rank_services(normal, 0, 0, [service(parts={})])
    assert len(result) == 1
    assert result[0]["parts"] == {}
    assert "无配件库存要求" in result[0]["explanation"]


@pytest.mark.parametrize(
    ("latitude", "longitude"), [(91, 0), (0, 181), (float("inf"), 0), (0, "121"), (True, 0)]
)
def test_invalid_customer_coordinates(attribution, latitude, longitude):
    with pytest.raises(DomainError) as error:
        rank_services(attribution, latitude, longitude)
    assert error.value.code == "invalid_coordinates"


def test_duplicate_service_ids_rejected(attribution):
    with pytest.raises(DomainError) as error:
        rank_services(attribution, 0, 0, [service(), service()])
    assert error.value.code == "duplicate_service"
