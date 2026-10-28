"""Real model training/inference, artifact integrity, grouping and routing tests."""

import json
import math
import shutil

import pytest

from intelligent_diagnosis.features import FEATURE_COLUMNS, extract_features
from intelligent_diagnosis.ml import DiagnosisModels, train_models
from intelligent_diagnosis.routing import classify_complaint
from intelligent_diagnosis.schemas import Domain, DomainError
from intelligent_diagnosis.synthetic import CLASS_DOMAIN, CLASSES, COMPLAINTS, make_request


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("real-lightgbm")
    metadata = train_models(directory, seed=17, n_devices=48)
    return directory, metadata, DiagnosisModels(directory, metadata["version"])


def _feedback_record(class_id="battery_depleted", device_id="expert-new-device", seed=81):
    request = make_request(class_id, seed=seed, device_id=device_id)
    return {
        "event_id": f"feedback-event-{seed}", "device_id": device_id,
        "request": request.model_dump(mode="json"), "confirmed_class": class_id,
        "feedback_id": f"expert-feedback-{seed}",
    }


def test_activation_metadata_aliases_match_actual_frozen_metrics(trained):
    _, metadata, _ = trained
    assert len(metadata["evaluation_fingerprint"]) == 64
    for domain in Domain:
        info = metadata["domains"][domain.value]
        assert info["test_macro_f1"] == metadata["metrics"][domain.value]["test"]["macro_f1"]
        assert info["test_accuracy"] == metadata["metrics"][domain.value]["test"]["accuracy"]
        assert info["feedback_train_events"] == 0


def test_feedback_really_enters_train_without_changing_frozen_evaluation(trained, tmp_path):
    _, original, _ = trained
    record = _feedback_record()
    changed = train_models(tmp_path, seed=17, n_devices=48, feedback_records=[record])
    assert changed["feedback"]["accepted_count"] == 1
    assert changed["feedback"]["rejected_count"] == 0
    assert changed["feedback"]["partition"] == "train_only"
    assert changed["evaluation_fingerprint"] == original["evaluation_fingerprint"]
    assert changed["splits"]["test"] == original["splits"]["test"]
    assert changed["splits"]["validation"] == original["splits"]["validation"]
    assert changed["splits"]["train"]["n_events"] == original["splits"]["train"]["n_events"] + 1
    assert changed["splits"]["train"]["n_devices"] == original["splits"]["train"]["n_devices"] + 1
    assert "expert-new-device" in changed["splits"]["train"]["device_ids"]
    assert changed["domains"]["low_voltage"]["feedback_train_events"] == 1
    assert changed["domains"]["low_voltage"]["train_events"] == (
        original["domains"]["low_voltage"]["train_events"] + 1
    )
    assert changed["domains"]["low_voltage"]["model_sha256"] != (
        original["domains"]["low_voltage"]["model_sha256"]
    )
    assert changed["version"] != original["version"]
    assert DiagnosisModels(tmp_path, changed["version"]).metadata == changed


def test_feedback_excludes_frozen_devices_bad_windows_and_duplicate_events(trained, tmp_path):
    _, original, _ = trained
    held_out = _feedback_record(device_id=original["splits"]["test"]["device_ids"][0], seed=83)
    held_validation = _feedback_record(
        device_id=original["splits"]["validation"]["device_ids"][0], seed=84
    )
    valid = _feedback_record(seed=85)
    invalid = _feedback_record(seed=86)
    invalid["request"]["samples"][0]["values"].pop("voltage")
    metadata = train_models(
        tmp_path, seed=17, n_devices=48,
        feedback_records=[held_out, held_validation, valid, valid, invalid],
    )
    report = metadata["feedback"]
    assert report["accepted_count"] == 1
    assert report["rejected_count"] == 3
    assert report["duplicate_count"] == 1
    assert [row["reason"] for row in report["rejected"]].count("held_out_device") == 2
    assert metadata["evaluation_fingerprint"] == original["evaluation_fingerprint"]
    assert not set(metadata["splits"]["train"]["device_ids"]) & set(
        metadata["splits"]["test"]["device_ids"]
    )


def test_conflicting_expert_labels_are_not_silently_overwritten(tmp_path):
    first = _feedback_record()
    conflicting = {**first, "confirmed_class": "charging_failure", "feedback_id": "feedback-other"}
    with pytest.raises(DomainError) as error:
        train_models(tmp_path, n_devices=24, feedback_records=[first, conflicting])
    assert error.value.code == "FEEDBACK_CONFLICT"
    assert not list(tmp_path.iterdir())


def test_cross_domain_expert_label_cannot_supply_training_features(tmp_path):
    record = _feedback_record()
    record["request"]["domain"] = "safety"
    with pytest.raises(DomainError) as error:
        train_models(tmp_path, n_devices=24, feedback_records=[record])
    assert error.value.code == "FEEDBACK_DOMAIN_MISMATCH"


def test_evaluation_fingerprint_changes_with_different_frozen_dataset(trained, tmp_path):
    _, original, _ = trained
    different = train_models(tmp_path, seed=18, n_devices=48)
    assert different["evaluation_fingerprint"] != original["evaluation_fingerprint"]


def test_device_and_event_split_cannot_leak(trained):
    _, metadata, _ = trained
    splits = metadata["splits"]
    groups = [set(splits[name]["device_ids"]) for name in ("train", "validation", "test")]
    assert all(groups)
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert len(set.union(*groups)) == 48
    for split in splits.values():
        assert split["n_devices"] == len(split["device_ids"])
        assert split["n_events"] == split["n_devices"] * 8 * 2


def test_models_are_real_multiclass_lightgbm_text_and_json(trained):
    directory, metadata, _ = trained
    artifacts = directory / metadata["version"]
    assert {path.suffix for path in artifacts.iterdir()} == {".json", ".txt"}
    for domain in Domain:
        text = (artifacts / f"{domain.value}.txt").read_text()
        assert "Tree=0" in text
        assert "num_class=4" in text
        assert "objective=multiclass" in text
        assert metadata["domains"][domain.value]["feature_columns"] == list(FEATURE_COLUMNS[domain])
    assert json.loads((artifacts / "metadata.json").read_text())["version"] == metadata["version"]


def test_metrics_report_frozen_test_baselines_and_validation_gate(trained):
    _, metadata, _ = trained
    for domain in Domain:
        metrics = metadata["metrics"][domain.value]
        assert 0 <= metrics["test"]["accuracy"] <= 1
        assert metrics["test"]["n_events"] == metadata["splits"]["test"]["n_devices"] * 8
        assert set(CLASSES[domain]) <= set(metrics["test"]["class_metrics"])
        assert set(metadata["baseline"][domain.value]) == {"majority_prior", "linear_logistic"}
        assert metadata["baseline"][domain.value]["majority_prior"]["accuracy"] == 0.25
        gate = metadata["threshold"][domain.value]
        assert gate["selection_partition"] == "validation"
        assert gate["calibrated_probability"] is False
        assert len(gate["selection_trials"]) > 1
        assert gate["validation_extreme_range_probe_rejection"] == 1.0
        assert len(metadata["domains"][domain.value]["candidate_validation"]) == 2
    assert any("production" in limitation for limitation in metadata["limitations"])


@pytest.mark.parametrize("class_id", tuple(CLASS_DOMAIN))
def test_predict_probabilities_are_model_outputs(trained, class_id):
    _, metadata, models = trained
    domain = CLASS_DOMAIN[class_id]
    request = make_request(class_id, seed=912, device_id="unseen-test-device")
    features, issues = extract_features(request, domain)
    assert not issues
    prediction = models.predict(domain, features)
    assert prediction.class_id == class_id
    assert prediction.domain == domain
    assert prediction.model_version == metadata["version"]
    assert set(prediction.scores) == set(CLASSES[domain])
    assert all(0 < value < 1 for value in prediction.scores.values())
    assert sum(prediction.scores.values()) == pytest.approx(1)
    # Acceptance can differ under validated support ranges; never assert 95%.
    if prediction.accepted:
        assert prediction.reliability_reason == "validation_gate_passed_uncalibrated"


def test_nan_uses_lightgbm_missing_branch_but_is_always_rejected(trained):
    _, _, models = trained
    features, _ = extract_features(make_request("circuit_open"), Domain.SAFETY)
    features["resistance_ohm_median"] = math.nan
    prediction = models.predict(Domain.SAFETY, features)
    assert not prediction.accepted
    assert "missing_features" in prediction.reliability_reason
    assert sum(prediction.scores.values()) == pytest.approx(1)


def test_extreme_ood_rejected_even_with_high_tree_confidence(trained):
    _, _, models = trained
    features, _ = extract_features(make_request("voltage_normal"), Domain.LOW_VOLTAGE)
    features["voltage_min"] = 10000.0
    prediction = models.predict(Domain.LOW_VOLTAGE, features)
    assert not prediction.accepted
    assert "out_of_distribution_range" in prediction.reliability_reason


def test_cross_domain_or_extra_feature_cannot_enter_model(trained):
    _, _, models = trained
    features, _ = extract_features(make_request("circuit_open"), Domain.SAFETY)
    with pytest.raises(DomainError) as wrong_domain:
        models.predict(Domain.LOW_VOLTAGE, features)
    assert wrong_domain.value.code == "FEATURE_SCHEMA_MISMATCH"
    features["class_id"] = 0.0
    with pytest.raises(DomainError):
        models.predict(Domain.SAFETY, features)


def test_infinite_feature_is_not_a_valid_probability_input(trained):
    _, _, models = trained
    features, _ = extract_features(make_request("circuit_open"), Domain.SAFETY)
    features["resistance_ohm_median"] = math.inf
    with pytest.raises(DomainError) as error:
        models.predict(Domain.SAFETY, features)
    assert error.value.code == "FEATURE_NONFINITE"


@pytest.mark.parametrize("version", ["missing", "../../etc", "sim-lgbm-" + "0" * 20])
def test_missing_or_path_traversal_version_is_domain_error(tmp_path, version):
    with pytest.raises(DomainError) as error:
        DiagnosisModels(tmp_path, version)
    assert error.value.code == "MODEL_NOT_FOUND"


@pytest.mark.parametrize("artifact", ["safety.txt", "metadata.json", "low_voltage.txt"])
def test_changed_artifacts_cannot_load(trained, tmp_path, artifact):
    directory, metadata, _ = trained
    version = metadata["version"]
    shutil.copytree(directory / version, tmp_path / version)
    path = tmp_path / version / artifact
    if artifact == "metadata.json":
        changed = json.loads(path.read_text())
        changed["domains"]["safety"]["feature_columns"].reverse()
        path.write_text(json.dumps(changed))
    else:
        path.write_text(path.read_text() + "\nmodified-content\n")
    with pytest.raises(DomainError) as error:
        DiagnosisModels(tmp_path, version)
    assert error.value.code == "MODEL_INVALID"


@pytest.mark.parametrize("content", ["null", "[]", "\"not metadata\"", "{broken"])
def test_malformed_metadata_is_domain_error(trained, tmp_path, content):
    directory, metadata, _ = trained
    version = metadata["version"]
    shutil.copytree(directory / version, tmp_path / version)
    (tmp_path / version / "metadata.json").write_text(content)
    with pytest.raises(DomainError) as error:
        DiagnosisModels(tmp_path, version)
    assert error.value.code == "MODEL_INVALID"


def test_missing_domain_model_never_falls_back(trained, tmp_path):
    directory, metadata, _ = trained
    version = metadata["version"]
    shutil.copytree(directory / version, tmp_path / version)
    (tmp_path / version / "low_voltage.txt").unlink()
    with pytest.raises(DomainError):
        DiagnosisModels(tmp_path, version)


def test_repeated_training_reuses_content_addressed_immutable_artifacts(trained):
    directory, metadata, _ = trained
    version_path = directory / metadata["version"]
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in version_path.iterdir()}
    repeated = train_models(directory, seed=17, n_devices=48)
    assert repeated == metadata
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in version_path.iterdir()} == before


def test_changed_existing_version_is_not_overwritten(trained, tmp_path):
    directory, metadata, _ = trained
    version = metadata["version"]
    shutil.copytree(directory / version, tmp_path / version)
    model = tmp_path / version / "safety.txt"
    model.write_text("do-not-overwrite")
    with pytest.raises(DomainError) as error:
        train_models(tmp_path, seed=17, n_devices=48)
    assert error.value.code == "MODEL_VERSION_CONFLICT"
    assert model.read_text() == "do-not-overwrite"


def test_too_few_devices_is_rejected_before_training(tmp_path):
    with pytest.raises(ValueError):
        train_models(tmp_path, n_devices=6)


@pytest.mark.parametrize("class_id", tuple(CLASS_DOMAIN))
def test_synthetic_complaints_route_to_the_correct_domain(class_id):
    for complaint in COMPLAINTS[class_id]:
        domain, score = classify_complaint(complaint)
        assert domain == CLASS_DOMAIN[class_id], (complaint, domain, score)
        assert 0 < score <= 1


@pytest.mark.parametrize("text,expected", [
    ("机器的安全联锁回路今天突然报警，请帮忙检查", Domain.SAFETY),
    ("安全传感器电阻跳变且保护灯报警", Domain.SAFETY),
    ("电瓶没电了，发动机启动时电压很低", Domain.LOW_VOLTAGE),
    ("运行过程中蓄电池充电电压不足", Domain.LOW_VOLTAGE),
])
def test_router_generalizes_to_original_held_out_phrases(text, expected):
    assert classify_complaint(text)[0] == expected


@pytest.mark.parametrize("text", [
    "", "你好", "安全", "电池", "设备坏了请帮我看看", "不知道", "今天下雨吗",
    "液压系统漏油了", "推广一下低压电源产品的优惠活动", "咨询安全传感器采购价格",
    "我要买蓄电池请报价", "想了解安全回路的培训课程", "电池故障检测设备促销广告",
    "安全回路是什么", "asdfqwerzzzz", "请帮我写诗", "发动机冷却水温过高",
    "想推广电池故障诊断服务", "请问安全联锁产品的价格是多少",
    "安全回路和蓄电池同时报警，不知道检查哪一个",
])
def test_router_abstains_on_weak_unknown_commercial_or_ambiguous_text(text):
    domain, confidence = classify_complaint(text)
    assert domain is None, (text, domain, confidence)
    assert 0 <= confidence <= 1
