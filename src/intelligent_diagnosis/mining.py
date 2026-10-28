"""按事件去重的真实共现统计；输出关联候选和待审树，不推断因果或发布。

将七个合成特征按知识库透明阈值离散，枚举单条件及不同特征的双条件
到已确认类别的关联规则。支持度分母是全部唯一事件（含缺失证据事件），
不是设备数；同设备的不同事件保留，重复上传同一事件不增加样本量。
"""

import json
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from math import isfinite
from numbers import Real

from .knowledge import CLASSES, KnowledgeBase
from .schemas import DomainError


def _matches(observed: object, condition: dict) -> bool:
    if not isinstance(observed, Real) or isinstance(observed, bool) or not isfinite(observed):
        return False
    operator, value = condition["operator"], condition["value"]
    if operator == "between":
        return value[0] <= observed <= value[1]
    if operator == "abs_ge":
        return abs(observed) >= value
    return {
        "lt": lambda: observed < value,
        "le": lambda: observed <= value,
        "gt": lambda: observed > value,
        "ge": lambda: observed >= value,
        "eq": lambda: observed == value,
    }[operator]()


def mine_patterns(records: list[dict], min_support: float = 0.1) -> list[dict]:
    """support=n(A且C)/N; confidence=n(A且C)/n(A); lift=confidence/(n(C)/N)。

    confirmed_class必须是已知类别，但此函数不代替上游确认权限校验。
    未知特征不用于构造条件；缺失值不视为条件成立，阈值仅为合成协议。
    tree_draft还须专家决定域/映射，合并为完整快照并经管理员审批。
    """
    if (
        not isinstance(min_support, Real)
        or isinstance(min_support, bool)
        or not isfinite(min_support)
        or not 0 < min_support <= 1
    ):
        raise DomainError("invalid_min_support", "最小支持度须在(0,1]内", 422)
    if not isinstance(records, list):
        raise DomainError("invalid_mining_records", "挖掘记录必须为列表", 422)
    seen: dict[str, str] = {}
    events: list[dict] = []
    known_classes = set().union(*CLASSES.values())
    for record in records:
        if (
            not isinstance(record, dict)
            or not {"event_id", "device_id", "features", "confirmed_class"} <= record.keys()
        ):
            raise DomainError("invalid_mining_record", "记录须包含事件、设备、特征和确认类别", 422)
        if (
            not isinstance(record["event_id"], str)
            or not record["event_id"].strip()
            or not isinstance(record["device_id"], str)
            or not record["device_id"].strip()
            or not isinstance(record["features"], dict)
            or any(not isinstance(key, str) for key in record["features"])
            or not isinstance(record["confirmed_class"], str)
            or record["confirmed_class"] not in known_classes
        ):
            raise DomainError("invalid_mining_record", "记录标识、特征字典或确认类别无效", 422)
        try:
            fingerprint = json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise DomainError("invalid_mining_record", "记录须为有限数值的JSON数据", 422) from exc
        event_id = record["event_id"]
        if event_id in seen:
            if seen[event_id] != fingerprint:
                raise DomainError(
                    "event_content_conflict", "同一事件标识对应不同内容，拒绝挖掘", 409
                )
            continue
        seen[event_id] = fingerprint
        events.append(record)
    if not events:
        return []

    knowledge = KnowledgeBase()
    conditions = knowledge.snapshot()["conditions"]
    antecedent_counts: Counter = Counter()
    class_counts: Counter = Counter()
    joint_counts: Counter = Counter()
    for record in events:
        observed = record["features"]
        items = sorted(
            key for key, rule in conditions.items() if _matches(observed.get(rule["feature"]), rule)
        )
        antecedents = [(item,) for item in items]
        antecedents.extend(
            pair
            for pair in combinations(items, 2)
            if conditions[pair[0]]["feature"] != conditions[pair[1]]["feature"]
        )
        label = record["confirmed_class"]
        class_counts[label] += 1
        for antecedent in antecedents:
            antecedent_counts[antecedent] += 1
            joint_counts[(antecedent, label)] += 1

    patterns: list[dict] = []
    total = len(events)
    for (antecedent, label), joint_count in joint_counts.items():
        support = joint_count / total
        if support < min_support:
            continue
        confidence = joint_count / antecedent_counts[antecedent]
        lift = confidence / (class_counts[label] / total)
        domain = next(domain for domain, labels in CLASSES.items() if label in labels)
        condition_details = [deepcopy(conditions[key]) for key in antecedent]
        digest = sha256(
            json.dumps([antecedent, label], separators=(",", ":")).encode()
        ).hexdigest()[:16]
        patterns.append(
            {
                "pattern_id": f"association-{digest}",
                "kind": "association_candidate",
                "antecedent": list(antecedent),
                "consequent": label,
                "support": support,
                "confidence": confidence,
                "lift": lift,
                "event_count": total,
                "joint_count": joint_count,
                "antecedent_count": antecedent_counts[antecedent],
                "consequent_count": class_counts[label],
                "condition_details": condition_details,
                "tree_draft": {
                    "status": "pending_review",
                    "base_knowledge_version": knowledge.version,
                    "domain": domain,
                    "target_class": label,
                    "relation": "AND",
                    "children": deepcopy(condition_details),
                    "requires_admin_approval": True,
                },
                "explanation": (
                    f"共{total}个唯一事件，条件出现{antecedent_counts[antecedent]}次，"
                    f"类别出现{class_counts[label]}次，共现{joint_count}次。"
                    "仅为合成阈值下的观察关联，不是因果、现实检修标准或自动发布结论；"
                    "同设备多事件可能相关，须人工复核样本偏差后审批完整知识快照。"
                ),
            }
        )
    return sorted(
        patterns,
        key=lambda item: (
            -item["support"],
            -item["confidence"],
            -item["lift"],
            tuple(item["antecedent"]),
            item["consequent"],
        ),
    )
