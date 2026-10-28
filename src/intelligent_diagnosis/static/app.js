"use strict";

(() => {
  // 令牌仅存在于此闭包内；不写浏览器持久存储，不放在URL中。
  let token = "";
  let actor = null;
  let selected = null;
  let session = 0;
  const $ = (id) => document.getElementById(id);
  const pretty = (value) => JSON.stringify(value, null, 2);
  const newId = (prefix) => `${prefix}-${crypto.randomUUID()}`;
  const outcomeNames = {unknown: "未知 / 待补充", review_required: "必须人工复核", resolved: "合成规则满足 / 非维修完成"};
  const stageNames = {perception: "感知", routing: "领域路由", execution: "模型与证据执行", gating: "门控", feedback_pending: "等待反馈"};
  const evidenceNames = {supported: "支持", refuted: "反驳", missing: "缺失"};

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }

  function notice(message, failed = false) {
    $("notice").textContent = message;
    $("notice").className = failed ? "error" : "";
  }

  function clearSession() {
    session += 1;
    token = "";
    actor = null;
    selected = null;
    $("token-input").value = "";
    $("workspace").hidden = true;
    $("identity").textContent = "尚未登录";
    $("workspace").querySelectorAll("form").forEach((form) => form.reset());
    ["diagnosis-list", "audit-list", "result-meta", "stages", "probabilities", "evidence-list", "candidate-list", "service-id"].forEach((id) => $(id).replaceChildren());
    ["result-json", "result-reason", "model-gate", "attribution", "operation-result"].forEach((id) => { $(id).textContent = ""; });
    $("result-content").hidden = true;
    $("result-empty").hidden = false;
    $("outcome").textContent = "尚未选择";
    $("outcome").className = "badge";
  }

  async function api(path, options = {}) {
    if (!token) throw new Error("请先手动输入本地令牌登录。");
    const generation = session;
    const response = await fetch(path, {
      method: options.method || "GET",
      headers: {
        Authorization: `Bearer ${token}`,
        ...(options.body !== undefined ? {"Content-Type": "application/json"} : {}),
      },
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
      credentials: "omit",
      cache: "no-store",
      redirect: "error",
    });
    if (generation !== session) throw new Error("身份已更改，已忽略此前请求。");
    const data = await response.json();
    if (generation !== session) throw new Error("身份已更改，已忽略此前请求。");
    if (response.status === 401) clearSession();
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((item) => `${(item.loc || []).join(".")}: ${item.msg}`).join("；")
        : `请求未完成（HTTP ${response.status}）`;
      throw new Error(data.error ? `${data.error.code}：${data.error.message}` : detail);
    }
    return data;
  }

  async function perform(button, action) {
    if (button) button.disabled = true;
    try {
      await action();
    } catch (error) {
      notice(error instanceof Error ? error.message : "操作未完成，请检查输入。", true);
    } finally {
      if (button) button.disabled = false;
    }
  }

  function onClick(id, action) {
    $(id).addEventListener("click", (event) => perform(event.currentTarget, action));
  }

  function onSubmit(id, action) {
    $(id).addEventListener("submit", (event) => {
      event.preventDefault();
      perform(event.submitter, action);
    });
  }

  function parseEditor(id) {
    try {
      const value = JSON.parse($(id).value);
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error();
      // JSON.parse会把1e999解析为Infinity，不能再由stringify悄悄改为null。
      JSON.stringify(value, (key, item) => {
        if (typeof item === "number" && !Number.isFinite(item)) throw new Error();
        return item;
      });
      return value;
    } catch {
      throw new Error("请输入合法的JSON对象；不接受NaN、Infinity或超出有限范围的数值。");
    }
  }

  function showOperation(data) {
    $("operation-result").textContent = pretty(data);
    notice("操作已完成，响应已记录在页面底部。业务状态以响应为准。");
  }

  async function loadRecords() {
    const results = await api("/api/diagnoses");
    const list = $("diagnosis-list");
    list.replaceChildren();
    if (!results.length) list.append(node("li", "暂无诊断。加载左侧样例即可开始。", "hint"));
    results.forEach((result) => {
      const item = node("li");
      const button = node("button", undefined, "record-button");
      button.type = "button";
      button.append(node("strong", result.device_id), node("span", outcomeNames[result.outcome] || result.outcome), node("small", `${result.diagnosis_id} · ${result.created_at}`));
      button.addEventListener("click", () => perform(button, async () => {
        renderResult(await api(`/api/diagnoses/${encodeURIComponent(result.diagnosis_id)}`));
        $("result").scrollIntoView({block: "start"});
      }));
      item.append(button);
      list.append(item);
    });
  }

  function renderResult(result) {
    selected = result;
    $("result-empty").hidden = true;
    $("result-content").hidden = false;
    $("outcome").textContent = outcomeNames[result.outcome] || result.outcome;
    $("outcome").className = "badge";
    if (["unknown", "review_required", "resolved"].includes(result.outcome)) $("outcome").classList.add(result.outcome);
    $("result-meta").replaceChildren();
    Object.entries({"诊断ID": result.diagnosis_id, "请求ID": result.request_id, "设备": result.device_id, "入口": result.entry, "领域": result.domain || "尚未明确", "知识版本": result.knowledge_version, "模型版本": result.model_version || "未执行模型", "创建时间": result.created_at}).forEach(([key, value]) => {
      $("result-meta").append(node("dt", key), node("dd", value));
    });
    $("result-reason").textContent = `门控说明：${result.reason}`;
    $("stages").replaceChildren(...result.stages.map((stage) => node("li", stageNames[stage] || stage)));
    const prediction = result.prediction;
    $("model-gate").textContent = prediction
      ? `预测类别：${prediction.class_id}；模型可靠性门限：${prediction.accepted ? "通过" : "未通过"}；${prediction.reliability_reason}`
      : "未生成模型预测；不能据此认定设备正常。";
    $("probabilities").replaceChildren();
    if (prediction) {
      Object.entries(prediction.scores).sort((left, right) => right[1] - left[1]).forEach(([classId, value]) => {
        const row = node("tr");
        const cell = node("td");
        const progress = node("progress");
        progress.max = 1;
        progress.value = value;
        progress.setAttribute("aria-label", `${classId}预测概率`);
        cell.append(progress);
        row.append(node("td", classId), node("td", `${(value * 100).toFixed(2)}%`), cell);
        $("probabilities").append(row);
      });
    }
    $("evidence-list").replaceChildren();
    if (!result.evidence.length) $("evidence-list").append(node("li", "尚无可展示的证据检查；这不代表证据通过。", "hint"));
    result.evidence.forEach((check) => {
      const item = node("li");
      const state = Object.hasOwn(evidenceNames, check.status) ? check.status : "missing";
      item.append(node("span", evidenceNames[state], `badge ${state}`), node("strong", check.condition === "root" ? "root / 聚合证据" : check.condition), node("p", check.description), node("small", check.observed === null ? "无单一观测值；以证据状态及条件描述为准。" : `观测值：${check.observed}`));
      $("evidence-list").append(item);
    });
    $("attribution").textContent = result.attribution
      ? `${result.attribution.name}（${result.attribution.attribution_id}） · ${result.attribution.solution}${result.attribution.safety_critical ? " · 安全相关，必须专家复核" : ""}`
      : "尚未形成可确认归因，请补充信号或交由专家复核。";
    $("candidate-list").replaceChildren();
    $("service-id").replaceChildren();
    if (!result.dispatch_candidates.length) $("candidate-list").append(node("li", "没有可派单候选，不创建虚构服务确认。", "hint"));
    result.dispatch_candidates.forEach((candidate, index) => {
      const item = node("li", undefined, "subcard");
      item.append(node("h4", `${index + 1}. ${candidate.name}（${candidate.service_id}）`), node("p", candidate.explanation), node("small", `候选综合得分：${candidate.score}；球面距离：${candidate.distance_km} km`));
      $("candidate-list").append(item);
      const option = node("option", `${candidate.name} / ${candidate.service_id}`);
      option.value = candidate.service_id;
      $("service-id").append(option);
    });
    $("assign-submit").disabled = !result.dispatch_candidates.length;
    $("feedback-form").reset();
    $("feedback-id").value = newId("feedback");
    $("assign-id").value = newId("assign");
    $("close-id").value = newId("close");
    $("correct-class").value = prediction ? prediction.class_id : "";
    $("result-json").textContent = pretty(result);
    notice("已展示实际服务结果。未知、证据缺失与人工复核不是诊断成功或维修完成。");
  }

  function selectedPath(action) {
    if (!selected) throw new Error("请先选择诊断记录。");
    return `/api/diagnoses/${encodeURIComponent(selected.diagnosis_id)}/${action}`;
  }

  onSubmit("login-form", async () => {
    const entered = $("token-input").value.trim();
    clearSession();
    if (!entered) throw new Error("请输入本地令牌。");
    token = entered;
    const generation = session;
    try {
      actor = await api("/api/me");
    } catch (error) {
      if (generation === session) clearSession();
      throw error;
    }
    $("identity").textContent = `${actor.actor_id} / ${actor.role}`;
    $("workspace").hidden = false;
    const isExpert = ["expert", "admin"].includes(actor.role);
    document.querySelectorAll(".expert-only").forEach((element) => { element.hidden = !isExpert; });
    document.querySelectorAll(".admin-only").forEach((element) => { element.hidden = actor.role !== "admin"; });
    $("knowledge-json").readOnly = !isExpert;
    $("activate-id").value = newId("activate");
    $("rollback-id").value = newId("rollback");
    notice("登录成功。令牌只保留在当前页面内存；退出或刷新后清除。");
    await loadRecords();
  });

  onClick("logout", async () => {
    clearSession();
    notice("已清除内存令牌与当前身份数据。刷新页面也会退出。");
  });

  onClick("load-example", async () => {
    const example = await api(`/api/examples/${encodeURIComponent($("example-class").value)}`);
    $("request-json").value = pretty(example);
    notice("已生成新的原创合成样例与请求ID。可编辑JSON后提交；这不是现场故障记录。");
  });

  onSubmit("diagnosis-form", async () => {
    const result = await api("/api/diagnoses", {method: "POST", body: parseEditor("request-json")});
    renderResult(result);
    await loadRecords();
  });

  onClick("prepare-telemetry", async () => {
    const request = parseEditor("request-json");
    const safetyClasses = ["circuit_open", "circuit_short", "safety_sensor_invalid", "safety_normal"];
    $("telemetry-json").value = pretty({
      event_id: newId("telemetry"), device_id: request.device_id,
      domain: request.domain || (safetyClasses.includes($("example-class").value) ? "safety" : "low_voltage"),
      samples: request.samples, latitude: request.latitude, longitude: request.longitude,
    });
    notice("遥测草稿已准备；领域默认参考当前场景选择，请核对JSON再提交。不会修改告警值来保证触发。");
  });

  onSubmit("telemetry-form", async () => {
    const response = await api("/api/telemetry", {method: "POST", body: parseEditor("telemetry-json")});
    if (response.diagnosis) renderResult(response.diagnosis);
    showOperation(response);
    notice(`遥测处理：${response.status} / ${response.reason}；生成${response.trigger_count}条诊断。多条时上方展示首条，全部结果见操作响应和记录列表。`);
    await loadRecords();
  });

  onClick("refresh-records", async () => {
    await loadRecords();
    notice("已刷新当前身份可见的诊断记录。");
  });

  onSubmit("feedback-form", async () => {
    showOperation(await api(selectedPath("feedback"), {method: "POST", body: {
      feedback_id: $("feedback-id").value.trim(), correct_class: $("correct-class").value.trim(),
      actual_action: $("actual-action").value, repair_confirmed: $("repair-confirmed").checked,
      note: $("feedback-note").value,
    }}));
  });

  onSubmit("assign-form", async () => {
    showOperation(await api(selectedPath("assign"), {method: "POST", body: {
      service_id: $("service-id").value, action_id: $("assign-id").value.trim(),
    }}));
    notice("合成派单已确认；不表示已完成真实预约或维修。参见最近操作响应。");
  });

  onClick("load-knowledge", async () => {
    $("knowledge-json").value = pretty(await api("/api/knowledge"));
    notice("已加载当前已发布快照。提交草稿前必须使用新的version，不覆盖现有版本。");
  });

  onSubmit("knowledge-form", async () => {
    const response = await api("/api/knowledge/proposals", {method: "POST", body: parseEditor("knowledge-json")});
    $("publish-version").value = response.version;
    showOperation(response);
    notice("知识草稿已提交，尚未生效；请由不同身份的管理员独立批准。");
  });

  onSubmit("publish-form", async () => {
    showOperation(await api(`/api/knowledge/${encodeURIComponent($("publish-version").value.trim())}/publish`, {method: "POST"}));
  });

  onSubmit("activate-form", async () => {
    showOperation(await api("/api/models/activate", {method: "POST", body: {
      version: $("model-version").value.trim(), action_id: $("activate-id").value.trim(),
    }}));
  });

  onSubmit("close-form", async () => {
    showOperation(await api(selectedPath("assignment/close"), {method: "POST", body: {
      action_id: $("close-id").value.trim(), outcome: $("close-outcome").value, note: $("close-note").value,
    }}));
  });

  onSubmit("rollback-form", async () => {
    showOperation(await api("/api/models/rollback", {method: "POST", body: {
      version: $("rollback-version").value.trim(), action_id: $("rollback-id").value.trim(), reason: $("rollback-reason").value,
    }}));
  });

  onClick("refresh-audit", async () => {
    const rows = await api("/api/audit");
    $("audit-list").replaceChildren();
    if (!rows.length) $("audit-list").append(node("li", "暂无审计事件。", "hint"));
    rows.forEach((row) => {
      const item = node("li");
      item.append(node("strong", `#${row.seq} · ${row.event}`), node("p", `${row.at} · ${row.actor}`), node("pre", row.payload));
      $("audit-list").append(item);
    });
    notice("审计已刷新；字段均按纯文本展示。");
  });
})();
