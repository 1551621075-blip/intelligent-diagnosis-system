"""原创合成诊断知识图：模型类别、业务归因、方案为三个独立对象。

阈值仅对应本项目合成数据协议，不是实车标准。图的AND/OR采用三值逻辑，
无观测是missing而不是反证；只有根条件supported表示证据支持该合成类别。
快照构建只做校验，不代表管理员审批、发布或对现实故障的因果证明。
"""

import re
from copy import deepcopy
from math import isfinite
from numbers import Real

import networkx as nx

from .schemas import Attribution, Domain, DomainError, EvidenceCheck

FEATURES = frozenset(
    {
        "resistance_ohm_median",
        "supply_voltage_min",
        "warning_active_max",
        "voltage_min",
        "current_a_mean",
        "engine_running_max",
        "terminal_drop_v_max",
    }
)
CLASSES = {
    "safety": {"circuit_open", "circuit_short", "safety_sensor_invalid", "safety_normal"},
    "low_voltage": {
        "battery_depleted",
        "charging_failure",
        "terminal_resistance",
        "voltage_normal",
    },
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")
_VERSION = re.compile(r"^knowledge-v[1-9][0-9]*$")
_OPERATORS = {"lt", "le", "gt", "ge", "eq", "between", "abs_ge"}


def _fail(message: str) -> None:
    raise DomainError("invalid_knowledge", message, 422)


def _identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fields(value: object, fields: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{label}字段不完整或存在未知字段")


def _domain(domain: Domain | str) -> str:
    value = "low_voltage" if domain == "lv" else domain
    if not isinstance(value, str) or value not in CLASSES:
        raise DomainError("unknown_domain", "未识别的诊断域", 422)
    return str(value)


def _default_snapshot(version: str) -> dict:
    conditions: dict = {}
    relations: dict = {}
    classes: dict = {domain: {} for domain in CLASSES}
    attributions: dict = {}
    solutions: dict = {}

    def condition(key: str, feature: str, operator: str, value: object, description: str) -> str:
        conditions[key] = {
            "feature": feature,
            "operator": operator,
            "value": value,
            "description": f"合成协议：{description}（非现实检修标准）",
        }
        return key

    def relation(key: str, operator: str, children: list[str], description: str) -> str:
        relations[key] = {"relation": operator, "children": children, "description": description}
        return key

    def mapping(
        domain: str,
        class_id: str,
        suffix: str,
        name: str,
        root: str,
        solution: str,
        skill: str,
        part: str | None = None,
        critical: bool = False,
    ) -> None:
        attribution_id, solution_id = f"cause-{suffix}", f"plan-{suffix}"
        classes[domain][class_id] = {"name": name, "attribution_id": attribution_id, "root": root}
        attributions[attribution_id] = {
            "name": name,
            "solution_id": solution_id,
            "required_skill": skill,
            "required_part": part,
            "safety_critical": critical,
        }
        solutions[solution_id] = {"description": solution}

    supply_ok = condition("supply-ok", "supply_voltage_min", "ge", 10.0, "安全支路供电不低于10伏")
    warning = condition("warning-on", "warning_active_max", "eq", 1.0, "安全告警已激活")
    no_warning = condition("warning-off", "warning_active_max", "eq", 0.0, "安全告警未激活")
    open_r = condition(
        "resistance-open", "resistance_ohm_median", "ge", 1000.0, "支路电阻中位数不低于1000欧"
    )
    short_r = condition(
        "resistance-short", "resistance_ohm_median", "le", 10.0, "支路电阻中位数不高于10欧"
    )
    normal_r = condition(
        "resistance-normal",
        "resistance_ohm_median",
        "between",
        [10.0, 1000.0],
        "支路电阻中位数在10至1000欧之间",
    )
    bad_supply = condition(
        "supply-invalid", "supply_voltage_min", "lt", 10.0, "安全支路供电低于10伏，观测链需核查"
    )
    root_open = relation(
        "rule-open", "AND", [open_r, supply_ok, warning], "开路形态须同时满足三项合成证据"
    )
    root_short = relation(
        "rule-short", "AND", [short_r, supply_ok, warning], "短路形态须同时满足三项合成证据"
    )
    root_sensor = relation(
        "rule-sensor", "AND", [bad_supply, warning], "传感异常形态须有供电与告警证据"
    )
    root_safety_normal = relation(
        "rule-safety-normal",
        "AND",
        [normal_r, supply_ok, no_warning],
        "安全支路合成正常窗口须三项同时成立",
    )
    safety_plan = (
        "仅合成诊断建议：停止自动闭环，交由具备安全支路技能的人员复核；不得据此宣称已维修。"
    )
    mapping(
        "safety",
        "circuit_open",
        "s01",
        "安全支路断续归因",
        root_open,
        safety_plan + "复核连接连续性后再制定作业方案。",
        "safety_loop",
        "loop_harness",
        True,
    )
    mapping(
        "safety",
        "circuit_short",
        "s02",
        "安全支路低阻归因",
        root_short,
        safety_plan + "复核异常低阻路径后再制定作业方案。",
        "safety_loop",
        "loop_harness",
        True,
    )
    mapping(
        "safety",
        "safety_sensor_invalid",
        "s03",
        "安全观测链异常归因",
        root_sensor,
        safety_plan + "先核查观测链供电与传感完整性。",
        "safety_loop",
        "safety_sensor",
        True,
    )
    mapping(
        "safety",
        "safety_normal",
        "s00",
        "安全支路合成正常",
        root_safety_normal,
        "当前证据符合合成正常窗口，无需据此安排维修；不代表现实安全认证。",
        "safety_loop",
    )

    stopped = condition("engine-stopped", "engine_running_max", "eq", 0.0, "发动机运行标记为0")
    running = condition("engine-running", "engine_running_max", "eq", 1.0, "发动机运行标记为1")
    depleted = condition("voltage-depleted", "voltage_min", "lt", 11.8, "低压电源最低值低于11.8伏")
    undercharge = condition(
        "voltage-undercharge", "voltage_min", "lt", 13.0, "运行时低压电源最低值低于13伏"
    )
    drop_high = condition("drop-high", "terminal_drop_v_max", "gt", 0.5, "端子压降峰值高于0.5伏")
    loaded = condition("current-loaded", "current_a_mean", "abs_ge", 5.0, "平均电流绝对值不低于5安")
    drop_normal = condition(
        "drop-normal", "terminal_drop_v_max", "between", [0.0, 0.3], "端子压降峰值在0至0.3伏之间"
    )
    voltage_normal = condition(
        "voltage-normal",
        "voltage_min",
        "between",
        [12.4, 15.0],
        "静止低压电源最低值在12.4至15伏之间",
    )
    voltage_running = condition(
        "voltage-running-normal",
        "voltage_min",
        "between",
        [13.0, 15.0],
        "运行低压电源最低值在13至15伏之间",
    )
    root_battery = relation(
        "rule-battery",
        "AND",
        [depleted, stopped, drop_normal],
        "亏电合成证据要求低电压、静止和正常端子压降",
    )
    root_charging = relation(
        "rule-charging",
        "AND",
        [undercharge, running, drop_normal],
        "充电异常合成证据要求运行低电压与正常端子压降",
    )
    root_terminal = relation(
        "rule-terminal", "AND", [drop_high, loaded], "端子异常合成证据要求压降与负载同时成立"
    )
    idle_normal = relation("rule-idle-normal", "AND", [stopped, voltage_normal], "静止正常电压分支")
    running_normal = relation(
        "rule-running-normal", "AND", [running, voltage_running], "运行正常电压分支"
    )
    voltage_branch = relation(
        "rule-voltage-branches",
        "OR",
        [idle_normal, running_normal],
        "静止或运行正常电压分支有一项成立",
    )
    root_lv_normal = relation(
        "rule-lv-normal",
        "AND",
        [voltage_branch, drop_normal],
        "正常电压分支和正常端子压降均须有证据",
    )
    lv_plan = "仅合成诊断建议：证据支持时可完成诊断闭环，但不等于故障已修复；"
    mapping(
        "low_voltage",
        "battery_depleted",
        "v01",
        "低压储能不足归因",
        root_battery,
        lv_plan + "由人员确认储能状态及补能或更换需求。",
        "low_voltage",
        "battery",
    )
    mapping(
        "low_voltage",
        "charging_failure",
        "v02",
        "低压补能异常归因",
        root_charging,
        lv_plan + "由人员复核补能回路后决定作业。",
        "low_voltage",
        "charging_unit",
    )
    mapping(
        "low_voltage",
        "terminal_resistance",
        "v03",
        "端子接触压降归因",
        root_terminal,
        lv_plan + "由人员核查端子接触与紧固状态。",
        "low_voltage",
        "terminal_kit",
    )
    mapping(
        "low_voltage",
        "voltage_normal",
        "v00",
        "低压电源合成正常",
        root_lv_normal,
        "当前证据符合合成正常窗口，可完成诊断闭环；无维修动作，不能据此声称已修复。",
        "low_voltage",
    )
    return {
        "version": version,
        "faults": {
            "safety": {"fault_id": "fault-safety-demo", "name": "合成安全支路告警"},
            "low_voltage": {"fault_id": "fault-voltage-demo", "name": "合成低压电源告警"},
        },
        "classes": classes,
        "attributions": attributions,
        "solutions": solutions,
        "conditions": conditions,
        "relations": relations,
    }


def _build_graph(snapshot: dict) -> nx.DiGraph:
    _fields(
        snapshot,
        {"version", "faults", "classes", "attributions", "solutions", "conditions", "relations"},
        "知识快照",
    )
    if not isinstance(snapshot["version"], str) or not _VERSION.fullmatch(snapshot["version"]):
        _fail("知识版本须为knowledge-v后接正整数")
    for key in ("faults", "classes", "attributions", "solutions", "conditions", "relations"):
        if not isinstance(snapshot[key], dict) or not snapshot[key]:
            _fail(f"{key}必须为非空映射")
    if set(snapshot["faults"]) != set(CLASSES) or set(snapshot["classes"]) != set(CLASSES):
        _fail("故障域必须完整且为safety及low_voltage")
    if (
        sum(len(snapshot[key]) for key in ("conditions", "relations", "attributions", "solutions"))
        > 2048
    ):
        _fail("知识图对象数量超过合成项目限制")
    graph = nx.DiGraph(version=snapshot["version"], protocol="synthetic-only")
    all_ids: set[str] = set()

    def add(node: tuple, identifier: str, **data: object) -> None:
        if not _identifier(identifier) or identifier in all_ids:
            _fail("对象标识必须合法且在类别、归因、方案、条件和关系间互不重复")
        all_ids.add(identifier)
        graph.add_node(node, **data)

    for domain, fault in snapshot["faults"].items():
        _fields(fault, {"fault_id", "name"}, "故障域")
        if not _text(fault["name"]):
            _fail("故障域名称不能为空")
        add(("domain", domain), fault["fault_id"], kind="domain", **fault)
    for solution_id, solution in snapshot["solutions"].items():
        _fields(solution, {"description"}, "方案")
        if not _text(solution["description"]):
            _fail("方案说明不能为空")
        add(("solution", solution_id), solution_id, kind="solution", **solution)
    for attribution_id, attribution in snapshot["attributions"].items():
        _fields(
            attribution,
            {"name", "solution_id", "required_skill", "required_part", "safety_critical"},
            "归因",
        )
        if (
            not _text(attribution["name"])
            or not _identifier(attribution["required_skill"])
            or (
                attribution["required_part"] is not None
                and not _identifier(attribution["required_part"])
            )
            or not isinstance(attribution["safety_critical"], bool)
            or not _identifier(attribution["solution_id"])
            or attribution["solution_id"] not in snapshot["solutions"]
        ):
            _fail("归因必须包含合法技能、配件、安全标记及有效方案引用")
        add(("attribution", attribution_id), attribution_id, kind="attribution", **attribution)
        graph.add_edge(
            ("attribution", attribution_id),
            ("solution", attribution["solution_id"]),
            kind="solution",
        )
    for condition_id, condition in snapshot["conditions"].items():
        _fields(condition, {"feature", "operator", "value", "description"}, "条件")
        if (
            not isinstance(condition["feature"], str)
            or condition["feature"] not in FEATURES
            or not isinstance(condition["operator"], str)
            or condition["operator"] not in _OPERATORS
            or not _text(condition["description"])
        ):
            _fail("条件须引用已知合成特征、支持的运算符并提供说明")
        value = condition["value"]
        if condition["operator"] == "between":
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not all(_number(v) for v in value)
                or value[0] > value[1]
            ):
                _fail("between条件须提供升序有限数值区间")
        elif not _number(value) or (condition["operator"] == "abs_ge" and value < 0):
            _fail("条件阈值须为有限数值，绝对值阈值不得为负")
        add(("condition", condition_id), condition_id, kind="condition", **condition)
    for relation_id, relation in snapshot["relations"].items():
        _fields(relation, {"relation", "children", "description"}, "关系")
        if (
            relation["relation"] not in ("AND", "OR")
            or not _text(relation["description"])
            or not isinstance(relation["children"], list)
            or not relation["children"]
            or any(not _identifier(child) for child in relation["children"])
            or len(set(relation["children"])) != len(relation["children"])
        ):
            _fail("关系必须为AND/OR，子项须非空且不可重复")
        add(("relation", relation_id), relation_id, kind="relation", **relation)
    for relation_id, relation in snapshot["relations"].items():
        for child in relation["children"]:
            if child in snapshot["conditions"]:
                kind = "condition"
            elif child in snapshot["relations"]:
                kind = "relation"
            else:
                _fail("关系引用了不存在的条件或关系")
            graph.add_edge(("relation", relation_id), (kind, child), kind="evidence")
    for domain, mappings in snapshot["classes"].items():
        if not isinstance(mappings, dict) or set(mappings) != CLASSES[domain]:
            _fail("模型类别必须完整并与故障域匹配")
        for class_id, mapping in mappings.items():
            _fields(mapping, {"name", "attribution_id", "root"}, "模型类别映射")
            if (
                not _text(mapping["name"])
                or not _identifier(mapping["attribution_id"])
                or mapping["attribution_id"] not in snapshot["attributions"]
                or not _identifier(mapping["root"])
                or mapping["root"] not in snapshot["relations"]
            ):
                _fail("类别必须绑定有效业务归因及关系根")
            attribution = snapshot["attributions"][mapping["attribution_id"]]
            if (
                domain == "safety"
                and class_id != "safety_normal"
                and not attribution["safety_critical"]
            ):
                _fail("安全域异常类必须保持安全关键标记")
            node = ("class", domain, class_id)
            add(node, class_id, kind="class", **mapping)
            graph.add_edge(("domain", domain), node, kind="class")
            graph.add_edge(node, ("attribution", mapping["attribution_id"]), kind="attribution")
            graph.add_edge(node, ("relation", mapping["root"]), kind="evidence")
    if not nx.is_directed_acyclic_graph(graph):
        _fail("知识图包含环，不能构建或发布")
    return nx.freeze(graph)


class KnowledgeBase:
    def __init__(self, version: str = "knowledge-v1"):
        self._snapshot = _default_snapshot(version)
        self.graph = _build_graph(self._snapshot)

    @property
    def version(self) -> str:
        return self._snapshot["version"]

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "KnowledgeBase":
        """完整校验后创建独立实例；合法仅表示可供审核，不会自动发布。"""
        data = deepcopy(snapshot)
        graph = _build_graph(data)
        instance = cls.__new__(cls)
        instance._snapshot = data
        instance.graph = graph
        return instance

    def snapshot(self) -> dict:
        return deepcopy(self._snapshot)

    def validate(self) -> None:
        _build_graph(self._snapshot)

    def fault_id(self, domain: Domain | str) -> str:
        return self._snapshot["faults"][_domain(domain)]["fault_id"]

    def validate_fault(self, domain: Domain | str, fault_id: str) -> bool:
        try:
            return self.fault_id(domain) == fault_id
        except DomainError:
            return False

    def _mapping(self, domain: Domain | str, class_id: str) -> dict:
        domain_key = _domain(domain)
        if not isinstance(class_id, str) or class_id not in self._snapshot["classes"][domain_key]:
            raise DomainError("class_mapping_missing", "模型类别在该故障域没有业务归因映射", 422)
        return self._snapshot["classes"][domain_key][class_id]

    def resolve(self, domain: Domain | str, class_id: str) -> Attribution:
        mapping = self._mapping(domain, class_id)
        identifier = mapping["attribution_id"]
        attribution = self._snapshot["attributions"][identifier]
        solution = self._snapshot["solutions"][attribution["solution_id"]]
        return Attribution(
            attribution_id=identifier, **attribution, solution=solution["description"]
        )

    def verify(self, domain: Domain | str, class_id: str, features: dict) -> list[EvidenceCheck]:
        """首项condition='root'是判定契约，其余逐项解释；不可对OR叶项直接all。

        AND: 有refuted则refuted，否则有missing则missing，否则supported。
        OR: 有supported则supported，否则有missing则missing，否则refuted。
        missing/None/非有限值/非数值均不提供证据；不会用默认零填补。
        """
        mapping = self._mapping(domain, class_id)
        if not isinstance(features, dict):
            raise DomainError("invalid_features", "诊断特征必须为字典", 422)
        root = ("relation", mapping["root"])
        reachable = self.graph.subgraph({root} | nx.descendants(self.graph, root))
        checks: dict[tuple, EvidenceCheck] = {}
        for node in reversed(list(nx.topological_sort(reachable))):
            data = self.graph.nodes[node]
            if data["kind"] == "condition":
                value = features.get(data["feature"])
                if not _number(value):
                    checks[node] = EvidenceCheck(
                        condition=node[1],
                        status="missing",
                        description=data["description"] + "；缺少有效有限观测",
                    )
                    continue
                value = float(value)
                threshold, operator = data["value"], data["operator"]
                if operator == "between":
                    supported = threshold[0] <= value <= threshold[1]
                elif operator == "abs_ge":
                    supported = abs(value) >= threshold
                else:
                    supported = {
                        "lt": value < threshold,
                        "le": value <= threshold,
                        "gt": value > threshold,
                        "ge": value >= threshold,
                        "eq": value == threshold,
                    }[operator]
                checks[node] = EvidenceCheck(
                    condition=node[1],
                    status="supported" if supported else "refuted",
                    observed=value,
                    description=data["description"],
                )
            else:
                states = [checks[child].status for child in reachable.successors(node)]
                if data["relation"] == "AND":
                    status = (
                        "refuted"
                        if "refuted" in states
                        else "missing"
                        if "missing" in states
                        else "supported"
                    )
                else:
                    status = (
                        "supported"
                        if "supported" in states
                        else "missing"
                        if "missing" in states
                        else "refuted"
                    )
                checks[node] = EvidenceCheck(
                    condition="root" if node == root else node[1],
                    status=status,
                    description=f"{data['description']}；{data['relation']}三值合成证据汇总",
                )
        return [checks[node] for node in nx.dfs_preorder_nodes(reachable, root)]
