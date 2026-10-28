from copy import deepcopy

import pytest

from intelligent_diagnosis.mining import mine_patterns
from intelligent_diagnosis.schemas import DomainError


def record(event_id, features=None, label="battery_depleted", device_id="device-a"):
    return {
        "event_id": event_id,
        "device_id": device_id,
        "features": {} if features is None else features,
        "confirmed_class": label,
    }


def pattern(rules, antecedent, consequent="battery_depleted"):
    return next(
        rule
        for rule in rules
        if rule["antecedent"] == antecedent and rule["consequent"] == consequent
    )


def fixture_records():
    return [
        record("e1", {"voltage_min": 10, "engine_running_max": 0}),
        record("e2", {"voltage_min": 11, "engine_running_max": 0}),
        record("e3", {"voltage_min": 11, "engine_running_max": 1}, "charging_failure"),
        record("e4", {"voltage_min": 14, "engine_running_max": 1}, "voltage_normal"),
    ]


def test_actual_support_confidence_lift():
    rules = mine_patterns(fixture_records())
    rule = pattern(rules, ["voltage-depleted"])
    assert rule["event_count"] == 4
    assert rule["joint_count"] == 2
    assert rule["antecedent_count"] == 3
    assert rule["consequent_count"] == 2
    assert rule["support"] == 0.5
    assert rule["confidence"] == pytest.approx(2 / 3)
    assert rule["lift"] == pytest.approx(4 / 3)
    pair = pattern(rules, ["engine-stopped", "voltage-depleted"])
    assert pair["support"] == 0.5
    assert pair["confidence"] == 1
    assert pair["lift"] == 2


def test_duplicate_events_are_idempotent_even_with_reordered_keys():
    records = fixture_records()
    repeated = dict(reversed(list(records[0].items())))
    repeated["features"] = dict(reversed(list(repeated["features"].items())))
    assert mine_patterns(records) == mine_patterns(records + [repeated] * 5)


@pytest.mark.parametrize(
    "change",
    [
        {"features": {"voltage_min": 9}},
        {"device_id": "other-device"},
        {"confirmed_class": "charging_failure"},
        {"note": "同ID新增内容"},
    ],
)
def test_same_id_different_content_rejected(change):
    first = record("same", {"voltage_min": 10})
    second = deepcopy(first)
    second.update(change)
    with pytest.raises(DomainError) as error:
        mine_patterns([first, second])
    assert error.value.code == "event_content_conflict"
    assert error.value.status == 409


def test_device_is_not_event_deduplication_key():
    records = [record("one", {"voltage_min": 10}), record("two", {"voltage_min": 10})]
    assert pattern(mine_patterns(records), ["voltage-depleted"])["joint_count"] == 2


def test_missing_features_stay_in_event_denominator():
    records = fixture_records() + [record("e5", {"voltage_min": None}, "voltage_normal")]
    rule = pattern(mine_patterns(records), ["voltage-depleted"])
    assert rule["event_count"] == 5
    assert rule["support"] == 0.4
    assert rule["confidence"] == pytest.approx(2 / 3)
    assert rule["lift"] == pytest.approx((2 / 3) / (2 / 5))
    assert mine_patterns([record("empty")]) == []


def test_support_threshold_is_inclusive():
    records = fixture_records()
    assert pattern(mine_patterns(records, min_support=0.5), ["voltage-depleted"])
    assert not any(
        rule["antecedent"] == ["voltage-depleted"] and rule["consequent"] == "battery_depleted"
        for rule in mine_patterns(records, min_support=0.51)
    )


def test_only_candidates_and_review_tree_not_publication_or_causation():
    records = fixture_records()
    original = deepcopy(records)
    rules = mine_patterns(records)
    assert records == original
    for rule in rules:
        assert rule["kind"] == "association_candidate"
        assert "不是因果" in rule["explanation"]
        assert rule["tree_draft"]["status"] == "pending_review"
        assert rule["tree_draft"]["requires_admin_approval"] is True
        assert rule["tree_draft"]["relation"] == "AND"
        assert rule["tree_draft"]["target_class"] == rule["consequent"]
        assert all(
            {"feature", "operator", "value", "description"} == set(item)
            for item in rule["tree_draft"]["children"]
        )
        assert "published" not in rule


def test_deterministic_results_independent_of_record_order():
    assert mine_patterns(fixture_records()) == mine_patterns(list(reversed(fixture_records())))


def test_rules_can_have_lift_below_one_without_claiming_positive_association():
    records = [
        record("one", {"voltage_min": 10}),
        record("two", {"voltage_min": 10}, "voltage_normal"),
        record("three", {}, "battery_depleted"),
    ]
    rule = pattern(mine_patterns(records), ["voltage-depleted"])
    assert rule["confidence"] == 0.5
    assert rule["lift"] == 0.75


@pytest.mark.parametrize("value", [0, -0.1, 1.1, float("nan"), float("inf"), "0.1", True, None])
def test_invalid_support_rejected(value):
    with pytest.raises(DomainError) as error:
        mine_patterns([], value)
    assert error.value.code == "invalid_min_support"


@pytest.mark.parametrize(
    "records",
    [
        None,
        {},
        [{}],
        [record("", {})],
        [record("e1", {}, "not-a-class")],
        [record("e1", {"voltage_min": float("nan")})],
        [record("e1", {"voltage_min": float("inf")})],
        [record("e1", {1: 10})],
    ],
)
def test_invalid_records_rejected(records):
    with pytest.raises(DomainError):
        mine_patterns(records)


def test_no_records_no_rules():
    assert mine_patterns([]) == []
