"""Feature and synthetic-data contract tests; no production fixtures."""

import math

import pytest

from intelligent_diagnosis.features import FEATURE_COLUMNS, extract_features
from intelligent_diagnosis.knowledge import KnowledgeBase
from intelligent_diagnosis.schemas import Domain, Entry, SignalSample
from intelligent_diagnosis.synthetic import CLASS_DOMAIN, CLASSES, make_request


@pytest.mark.parametrize("class_id", tuple(CLASS_DOMAIN))
def test_synthetic_events_have_fixed_finite_feature_schema(class_id):
    request = make_request(class_id, seed=19)
    features, issues = extract_features(request, CLASS_DOMAIN[class_id])
    assert issues == []
    assert tuple(features) == FEATURE_COLUMNS[CLASS_DOMAIN[class_id]]
    assert all(math.isfinite(value) for value in features.values())
    assert len(request.samples) >= 5
    assert request.complaint
    assert class_id not in features
    assert class_id not in request.request_id
    assert class_id not in request.device_id


def test_reproducible_seed_and_telemetry_bindings():
    first = make_request("circuit_open", seed=8, entry=Entry.TELEMETRY)
    assert first == make_request("circuit_open", seed=8, entry=Entry.TELEMETRY)
    assert first.domain == Domain.SAFETY
    assert first.fault_id
    assert first != make_request("circuit_open", seed=9, entry=Entry.TELEMETRY)
    with pytest.raises(ValueError):
        make_request("unsupported")


@pytest.mark.parametrize("class_id", tuple(CLASS_DOMAIN))
def test_telemetry_generator_binds_original_knowledge_domain_fault(class_id):
    request = make_request(class_id, seed=31, entry=Entry.TELEMETRY)
    assert KnowledgeBase().validate_fault(request.domain, request.fault_id)
    user_request = make_request(class_id, seed=31)
    assert request.samples == user_request.samples
    assert request.complaint == user_request.complaint


def test_window_is_inclusive_and_ignores_outside_extremes():
    request = make_request("voltage_normal", seed=4)
    original, issues = extract_features(request, Domain.LOW_VOLTAGE)
    assert not issues
    outside = [
        SignalSample(timestamp_ms=request.fault_time_ms + offset, values={"voltage": -99999})
        for offset in (-1001, 1001)
    ]
    changed = request.model_copy(update={"samples": outside + list(reversed(request.samples))})
    assert extract_features(changed, Domain.LOW_VOLTAGE) == (original, [])
    request.samples[0].values["voltage"] = 1.0
    features, _ = extract_features(request, Domain.LOW_VOLTAGE)
    assert features["voltage_min"] == 1.0


def test_identical_duplicates_do_not_reweight_statistics_but_are_reported():
    request = make_request("circuit_open")
    expected, _ = extract_features(request, Domain.SAFETY)
    request.samples.extend([request.samples[0]] * 5)
    actual, issues = extract_features(request, Domain.SAFETY)
    assert actual == expected
    assert any(issue.startswith("duplicate_timestamp:") for issue in issues)


def test_conflicting_duplicate_marks_affected_features_unavailable():
    request = make_request("circuit_open")
    duplicate = request.samples[0].model_copy(deep=True)
    duplicate.values["resistance_ohm"] = 1.0
    request.samples.append(duplicate)
    features, issues = extract_features(request, Domain.SAFETY)
    assert math.isnan(features["resistance_ohm_median"])
    assert math.isfinite(features["supply_voltage_min"])
    assert any(issue.startswith("conflicting_timestamp:") for issue in issues)


@pytest.mark.parametrize("missing", [None, "absent"])
def test_missing_signal_is_nan_never_zero(missing):
    request = make_request("battery_depleted")
    if missing is None:
        request.samples[3].values["voltage"] = None
    else:
        del request.samples[3].values["voltage"]
    features, issues = extract_features(request, Domain.LOW_VOLTAGE)
    assert math.isnan(features["voltage_min"])
    assert math.isnan(features["voltage_mean"])
    assert math.isnan(features["voltage_slope_v_per_s"])
    assert math.isfinite(features["current_a_mean"])
    assert any(issue.startswith("missing_signal:voltage:") for issue in issues)


@pytest.mark.parametrize("nonfinite", [math.inf, -math.inf, math.nan])
def test_defensive_nonfinite_check_for_internal_unvalidated_objects(nonfinite):
    request = make_request("circuit_short")
    request.samples[2].values["resistance_ohm"] = nonfinite
    features, issues = extract_features(request, Domain.SAFETY)
    assert math.isnan(features["resistance_ohm_min"])
    assert any(issue.startswith("nonfinite_signal:resistance_ohm:") for issue in issues)


def test_overflowed_aggregate_is_reported_as_nan():
    request = make_request("circuit_open")
    request.samples[0].values["resistance_ohm"] = 1e308
    features, issues = extract_features(request, Domain.SAFETY)
    assert math.isnan(features["resistance_ohm_std"])
    assert "unavailable_feature:resistance_ohm_std" in issues


@pytest.mark.parametrize("indices", [[], [0], [0, 8], [3, 4, 5], [0, 1, 2]])
def test_minimum_points_and_coverage_are_required(indices):
    request = make_request("safety_normal")
    request.samples = [request.samples[index] for index in indices]
    features, issues = extract_features(request, Domain.SAFETY)
    assert issues
    if len(indices) < 3:
        assert "insufficient_unique_points" in issues
    if indices != [0, 8]:
        assert "insufficient_time_coverage" in issues
    if not indices:
        assert all(math.isnan(value) for value in features.values())


def test_three_well_distributed_points_are_enough():
    request = make_request("safety_normal")
    request.samples = [request.samples[index] for index in (0, 4, 8)]
    assert extract_features(request, Domain.SAFETY)[1] == []


def test_duplicate_timestamps_do_not_satisfy_three_point_minimum():
    request = make_request("safety_normal")
    request.samples = [request.samples[0]] * 10 + [request.samples[-1]] * 10
    assert "insufficient_unique_points" in extract_features(request, Domain.SAFETY)[1]


def test_wrong_telemetry_domain_is_an_issue():
    request = make_request("safety_normal", entry=Entry.TELEMETRY)
    assert "domain_mismatch" in extract_features(request, Domain.LOW_VOLTAGE)[1]


@pytest.mark.parametrize("signal,value", [("warning_active", 0.5), ("dtc", -1), ("dtc", 1.5)])
def test_invalid_discrete_signals_are_rejected(signal, value):
    request = make_request("safety_normal")
    request.samples[2].values[signal] = value
    _, issues = extract_features(request, Domain.SAFETY)
    assert any(issue.startswith(f"invalid_signal:{signal}:") for issue in issues)


def test_linear_voltage_slope_uses_seconds_and_sorted_times():
    request = make_request("voltage_normal")
    for sample in request.samples:
        seconds = (sample.timestamp_ms - request.fault_time_ms) / 1000
        sample.values["voltage"] = 12 + 0.4 * seconds
    request.samples.reverse()
    features, issues = extract_features(request, Domain.LOW_VOLTAGE)
    assert not issues
    assert features["voltage_slope_v_per_s"] == pytest.approx(0.4)
    assert features["voltage_min"] == pytest.approx(11.6)


def test_generic_dtc_codes_do_not_reveal_fault_class():
    code_sets = {}
    for class_id in CLASS_DOMAIN:
        code_sets[class_id] = {
            sample.values["dtc"]
            for seed in range(12)
            for sample in make_request(class_id, seed=seed).samples
        }
    assert all(codes == {0, 1, 2, 3, 4} for codes in code_sets.values())


def test_domain_feature_columns_are_explicit_and_distinct():
    assert set(FEATURE_COLUMNS) == set(CLASSES) == set(Domain)
    assert "resistance_ohm_median" in FEATURE_COLUMNS[Domain.SAFETY]
    assert "voltage_min" not in FEATURE_COLUMNS[Domain.SAFETY]
    assert "voltage_min" in FEATURE_COLUMNS[Domain.LOW_VOLTAGE]
    assert "resistance_ohm_median" not in FEATURE_COLUMNS[Domain.LOW_VOLTAGE]
