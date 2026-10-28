from copy import deepcopy
from itertools import product

import networkx as nx
import pytest

from intelligent_diagnosis.knowledge import CLASSES, KnowledgeBase
from intelligent_diagnosis.schemas import Domain, DomainError


@pytest.fixture
def knowledge():
    return KnowledgeBase()


def test_real_graph_and_three_distinct_business_objects(knowledge):
    assert isinstance(knowledge.graph, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(knowledge.graph)
    snapshot = knowledge.snapshot()
    for domain, labels in CLASSES.items():
        for label in labels:
            attribution = knowledge.resolve(domain, label)
            assert len({label, attribution.attribution_id, attribution.solution_id}) == 3
            assert knowledge.graph.has_edge(
                ("class", domain, label), ("attribution", attribution.attribution_id)
            )
            assert knowledge.graph.has_edge(
                ("attribution", attribution.attribution_id), ("solution", attribution.solution_id)
            )
            assert attribution.solution_id in snapshot["solutions"]
            assert bool(attribution.solution)
            assert attribution.safety_critical == (domain == "safety" and label != "safety_normal")
    assert knowledge.validate() is None
    assert knowledge.version == "knowledge-v1"


def test_domain_bound_faults_and_mappings(knowledge):
    assert knowledge.validate_fault(Domain.SAFETY, knowledge.fault_id("safety"))
    assert knowledge.validate_fault("lv", knowledge.fault_id(Domain.LOW_VOLTAGE))
    assert not knowledge.validate_fault("safety", knowledge.fault_id("low_voltage"))
    assert not knowledge.validate_fault("unknown", "anything")
    for domain, label in [
        ("safety", "battery_depleted"),
        ("low_voltage", "circuit_open"),
        ("safety", "not-a-class"),
    ]:
        with pytest.raises(DomainError, match="没有业务归因映射"):
            knowledge.resolve(domain, label)
        with pytest.raises(DomainError):
            knowledge.verify(domain, label, {})
    with pytest.raises(DomainError):
        knowledge.fault_id("unknown")


@pytest.mark.parametrize(
    ("domain", "label", "features"),
    [
        (
            "safety",
            "circuit_open",
            {"resistance_ohm_median": 12000, "supply_voltage_min": 12, "warning_active_max": 1},
        ),
        (
            "safety",
            "circuit_short",
            {"resistance_ohm_median": 0.5, "supply_voltage_min": 12, "warning_active_max": 1},
        ),
        ("safety", "safety_sensor_invalid", {"supply_voltage_min": 6, "warning_active_max": 1}),
        (
            "safety",
            "safety_normal",
            {"resistance_ohm_median": 100, "supply_voltage_min": 12, "warning_active_max": 0},
        ),
        (
            "low_voltage",
            "battery_depleted",
            {"voltage_min": 10.5, "engine_running_max": 0, "terminal_drop_v_max": 0.1},
        ),
        (
            "low_voltage",
            "charging_failure",
            {"voltage_min": 12, "engine_running_max": 1, "terminal_drop_v_max": 0.1},
        ),
        ("low_voltage", "terminal_resistance", {"terminal_drop_v_max": 0.8, "current_a_mean": -10}),
        (
            "low_voltage",
            "voltage_normal",
            {"voltage_min": 12.8, "engine_running_max": 0, "terminal_drop_v_max": 0.1},
        ),
        (
            "low_voltage",
            "voltage_normal",
            {"voltage_min": 14, "engine_running_max": 1, "terminal_drop_v_max": 0.1},
        ),
    ],
)
def test_synthetic_classes_have_supported_evidence(knowledge, domain, label, features):
    checks = knowledge.verify(domain, label, features)
    assert checks[0].condition == "root"
    assert checks[0].status == "supported"
    assert all(check.description for check in checks)


@pytest.mark.parametrize(
    "invalid", [None, float("nan"), float("inf"), -float("inf"), "12000", True, []]
)
def test_invalid_or_missing_observation_is_missing(knowledge, invalid):
    checks = knowledge.verify(
        "safety",
        "circuit_open",
        {
            "resistance_ohm_median": invalid,
            "supply_voltage_min": 12,
            "warning_active_max": 1,
        },
    )
    assert checks[0].status == "missing"
    missing = next(check for check in checks if check.condition == "resistance-open")
    assert missing.status == "missing"
    assert missing.observed is None


def test_missing_and_conflicting_evidence_are_distinct(knowledge):
    assert knowledge.verify("safety", "circuit_open", {})[0].status == "missing"
    checks = knowledge.verify(
        "safety",
        "circuit_open",
        {
            "resistance_ohm_median": 12000,
            "warning_active_max": 0,
        },
    )
    assert checks[0].status == "refuted"
    assert next(check for check in checks if check.condition == "supply-ok").status == "missing"
    assert next(check for check in checks if check.condition == "warning-on").status == "refuted"
    assert (
        knowledge.verify(
            "low_voltage",
            "voltage_normal",
            {
                "voltage_min": 12.8,
                "terminal_drop_v_max": 0.1,
            },
        )[0].status
        == "missing"
    )


@pytest.mark.parametrize("relation", ["AND", "OR"])
@pytest.mark.parametrize("states", list(product(["supported", "refuted", "missing"], repeat=2)))
def test_complete_three_value_truth_tables(knowledge, relation, states):
    snapshot = knowledge.snapshot()
    snapshot["relations"]["rule-open"].update(
        relation=relation, children=["resistance-open", "supply-ok"]
    )
    modified = KnowledgeBase.from_snapshot(snapshot)
    features = {}
    for state, feature, supported, refuted in [
        (states[0], "resistance_ohm_median", 10000, 100),
        (states[1], "supply_voltage_min", 12, 6),
    ]:
        if state != "missing":
            features[feature] = supported if state == "supported" else refuted
    if relation == "AND":
        expected = (
            "refuted" if "refuted" in states else "missing" if "missing" in states else "supported"
        )
    else:
        expected = (
            "supported"
            if "supported" in states
            else "missing"
            if "missing" in states
            else "refuted"
        )
    assert modified.verify("safety", "circuit_open", features)[0].status == expected


def test_supported_or_branch_does_not_require_refuted_branch(knowledge):
    checks = knowledge.verify(
        "low_voltage",
        "voltage_normal",
        {
            "voltage_min": 12.8,
            "engine_running_max": 0,
            "terminal_drop_v_max": 0.1,
        },
    )
    assert checks[0].status == "supported"
    assert any(check.status == "refuted" for check in checks)


def test_snapshot_is_independent_and_roundtrips(knowledge):
    snapshot = knowledge.snapshot()
    snapshot["version"] = "knowledge-v2"
    rebuilt = KnowledgeBase.from_snapshot(snapshot)
    snapshot["conditions"]["resistance-open"]["value"] = 2
    assert knowledge.version == "knowledge-v1"
    assert rebuilt.version == "knowledge-v2"
    assert rebuilt.snapshot()["conditions"]["resistance-open"]["value"] == 1000
    assert KnowledgeBase.from_snapshot(rebuilt.snapshot()).snapshot() == rebuilt.snapshot()
    resolved = knowledge.resolve("safety", "circuit_open")
    resolved.name = "修改返回对象不影响知识"
    assert knowledge.resolve("safety", "circuit_open").name != resolved.name


@pytest.mark.parametrize(
    "version", ["", "v2", "knowledge-v0", "knowledge-v-2", "knowledge-v01", None, 2]
)
def test_invalid_version_rejected(version, knowledge):
    snapshot = knowledge.snapshot()
    snapshot["version"] = version
    with pytest.raises(DomainError):
        KnowledgeBase.from_snapshot(snapshot)
    with pytest.raises(DomainError):
        KnowledgeBase(version=version)


def test_cycles_even_disconnected_are_rejected(knowledge):
    snapshot = knowledge.snapshot()
    snapshot["relations"]["rule-open"]["children"].append("rule-open")
    with pytest.raises(DomainError, match="包含环"):
        KnowledgeBase.from_snapshot(snapshot)
    snapshot = knowledge.snapshot()
    snapshot["relations"]["extra-a"] = {
        "relation": "AND",
        "children": ["extra-b"],
        "description": "未发布分支",
    }
    snapshot["relations"]["extra-b"] = {
        "relation": "OR",
        "children": ["extra-a"],
        "description": "未发布分支",
    }
    with pytest.raises(DomainError, match="包含环"):
        KnowledgeBase.from_snapshot(snapshot)


@pytest.mark.parametrize(
    ("section", "key", "field", "invalid"),
    [
        ("relations", "rule-open", "relation", "XOR"),
        ("relations", "rule-open", "children", []),
        ("relations", "rule-open", "children", ["unknown"]),
        ("relations", "rule-open", "children", ["supply-ok", "supply-ok"]),
        ("relations", "rule-open", "children", [[]]),
        ("conditions", "resistance-open", "feature", "unmeasured_feature"),
        ("conditions", "resistance-open", "operator", "exec"),
        ("conditions", "resistance-open", "value", float("nan")),
        ("conditions", "resistance-normal", "value", [100, 1]),
        ("conditions", "resistance-normal", "value", [0, float("inf")]),
        ("attributions", "cause-s01", "solution_id", "missing-plan"),
        ("attributions", "cause-s01", "safety_critical", False),
        ("attributions", "cause-s01", "safety_critical", "true"),
        ("solutions", "plan-s01", "description", ""),
    ],
)
def test_invalid_relations_conditions_and_objects_rejected(knowledge, section, key, field, invalid):
    snapshot = knowledge.snapshot()
    snapshot[section][key][field] = invalid
    with pytest.raises(DomainError):
        KnowledgeBase.from_snapshot(snapshot)


def test_missing_class_and_overlapping_object_keys_rejected(knowledge):
    snapshot = knowledge.snapshot()
    snapshot["classes"]["safety"].pop("circuit_open")
    with pytest.raises(DomainError):
        KnowledgeBase.from_snapshot(snapshot)
    snapshot = knowledge.snapshot()
    snapshot["classes"]["safety"]["circuit_open"]["attribution_id"] = "missing"
    with pytest.raises(DomainError):
        KnowledgeBase.from_snapshot(snapshot)
    snapshot = knowledge.snapshot()
    snapshot["solutions"]["cause-s01"] = deepcopy(snapshot["solutions"]["plan-s01"])
    with pytest.raises(DomainError, match="互不重复"):
        KnowledgeBase.from_snapshot(snapshot)
    snapshot = knowledge.snapshot()
    snapshot["classes"]["safety"]["circuit_open"]["root"] = "nonexistent-root"
    with pytest.raises(DomainError):
        KnowledgeBase.from_snapshot(snapshot)


@pytest.mark.parametrize(
    ("domain", "label"),
    [(domain, label) for domain, labels in CLASSES.items() for label in sorted(labels)],
)
@pytest.mark.parametrize("seed", [0, 303, 509, 701])
def test_original_synthetic_protocol_reference_events(knowledge, domain, label, seed):
    from intelligent_diagnosis.features import extract_features
    from intelligent_diagnosis.schemas import Entry
    from intelligent_diagnosis.synthetic import make_request

    features, issues = extract_features(
        make_request(label, seed=seed, entry=Entry.TELEMETRY), Domain(domain)
    )
    assert issues == []
    checks = knowledge.verify(domain, label, features)
    assert checks[0].condition == "root"
    assert checks[0].status == "supported"
    assert knowledge.verify(domain, label, {})[0].status == "missing"
    for other in CLASSES[domain] - {label}:
        assert knowledge.verify(domain, other, features)[0].status != "supported"
    if label == "voltage_normal":
        # 未选运行状态分支的反证不否定OR根；发布回归也必须遵循根契约。
        assert any(check.status == "refuted" for check in checks[1:])
