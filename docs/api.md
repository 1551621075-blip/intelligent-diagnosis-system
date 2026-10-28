# 本地工作台与 HTTP API

本工作台仅演示原创合成电气场景。没有真实设备接入、现实门店预约或现场有效性保证。模型预测概率不是物理置信度；`resolved` 只表示当前合成规则满足，不表示设备已维修。安全相关归因仍须专家复核。

## 工厂、路径与访问边界

```python
from pathlib import Path
from intelligent_diagnosis.api import create_app

app = create_app(root=Path("runtime"), model_dir=Path("models"))
```

`create_app(root: Path | None = None, model_dir: Path | None = None) -> FastAPI` 在调用时才初始化 `Store` 与 `DiagnosisService`。仅导入 `create_app` 不创建数据库、运行目录或模型目录；没有模块级 `app` 实例。显式路径参数优先于 `DIAG_RUNTIME_DIR`、`DIAG_MODEL_DIR`，两者缺省分别为当前工作目录下的 `runtime`、`models`。

CLI 应调用工厂传入路径，或用 `uvicorn intelligent_diagnosis.api:create_app --factory --host 127.0.0.1` 并设置环境变量。只绑定回环地址；部署步骤及训练、身份预置命令见项目 README。API 不提供匿名创建身份、领取令牌、模型上传或训练接口。

工作台资源固定放在 `src/intelligent_diagnosis/templates/`、`src/intelligent_diagnosis/static/`，随wheel安装，不依赖仓库外的路径。`GET /` 返回页面，`GET /static/app.js`、`GET /static/style.css` 返回静态资源。`GET /openapi.json` 提供 OpenAPI Schema；不加载需要外部 CDN 的 Swagger/ReDoc 页面。

## 认证与授权

所有 `/api/*` 端点均须请求头：

```http
Authorization: Bearer <本地管理员预置的令牌>
```

API 通过 `Store.authenticate` 查找服务端身份；角色不是客户端参数。查询字符串、Cookie、`X-Role`、`X-Actor-ID` 均不能认证或提升权限。输入模型禁止额外字段，不能在诊断、反馈或派单请求里添加 `actor` / `role`。知识快照中的数据也不会用来构造身份。

- `operator`：提交诊断/遥测、仅看自己的诊断、读取当前知识及合成样例。
- `expert`：另外可复核诊断、反馈、派单、查看审计、提案知识草稿。
- `admin`：另外可激活模型、发布知识；知识提案人与批准人必须是不同身份。

`GET /api/me` 只返回 `{"actor_id":"...","role":"operator|expert|admin"}`。页面使用密码输入框手动登录，校验后清空输入框，令牌只保留于 JavaScript 闭包内；不写 `localStorage`、`sessionStorage`、Cookie 或 URL。退出清空当前身份数据，刷新需重新登录。更换身份后，旧请求响应不会显示为新身份的数据。

不配置开放 CORS。响应包含 `nosniff`、禁止嵌套与同源 CSP；业务响应 `Cache-Control: no-store`。页面只用 `textContent` / 文本节点呈现返回值，包括审计、备注、门店名称和 JSON，不把输入当 HTML 执行。应用不记录令牌或回显 Authorization。

## 端点与返回

除非另有说明，成功状态码为 **200**，返回 JSON。重复成功请求可能返回之前的业务结果，不能靠 HTTP 200 推断诊断成功。

| 方法与路径 | 角色 | 请求体 | 返回 |
| --- | --- | --- | --- |
| `GET /health` | 无需认证 | 无 | `{"status":"ok","synthetic":true}`，不包含路径、身份或令牌；不是模型就绪证明 |
| `GET /` | 无需认证 | 无 | 当前工作台 HTML，不注入业务数据或令牌 |
| `GET /api/me` | 全部已认证 | 无 | 服务端 `Actor` |
| `GET /api/examples/{class_id}` | 全部已认证 | 无 | 可直接提交的 `DiagnosisRequest` |
| `GET /api/diagnoses` | 全部已认证 | 无 | 最近100条授权范围内的 `DiagnosisResult` 数组 |
| `POST /api/diagnoses` | 全部已认证 | `DiagnosisRequest` | 完整 `DiagnosisResult`，包括未知/待复核 |
| `GET /api/diagnoses/{diagnosis_id}` | 全部已认证 | 无 | 单条 `DiagnosisResult`；操作员访问他人记录与记录不存在均为404 |
| `POST /api/telemetry` | 全部已认证 | `TelemetryEvent` | `status`、`reason`、`trigger_count`、全部`diagnoses`及首条`diagnosis`（对象或 `null`） |
| `POST /api/diagnoses/{diagnosis_id}/feedback` | 专家/管理员 | `FeedbackRequest` | 反馈及 `badcase_type`，不是即时训练结果 |
| `POST /api/diagnoses/{diagnosis_id}/assign` | 专家/管理员 | `DispatchRequest` | 合成确认，含 `repair_completed: false` |
| `GET /api/audit` | 专家/管理员 | 无 | 最近200条，含 `seq`、`at`、`actor`、`event`、`payload`；`payload` 是 JSON 文本 |
| `POST /api/models/activate` | 管理员 | `ModelActivation` | `status`、`version`、`previous` |
| `POST /api/models/rollback` | 管理员 | version/action_id/reason | 回滚到历史已批准模型，含前后版本与原因 |
| `GET /api/feedback/export` | 专家/管理员 | 无 | 专家标签与原始请求；冲突标签返回409 |
| `POST /api/diagnoses/{diagnosis_id}/assignment/close` | 专家/管理员 | action_id/outcome/note | cancelled归还预留资源，completed只释放工位；均为合成状态 |
| `GET /api/knowledge` | 全部已认证 | 无 | 当前已发布知识的**裸 snapshot 对象** |
| `POST /api/knowledge/proposals` | 专家/管理员 | 裸 snapshot 对象 | `{"version":"...","status":"draft"}` |
| `POST /api/knowledge/{version}/publish` | 管理员且非提案人 | 无 | `{"version":"...","status":"published"}` |

### 合成样例与诊断输入

支持以下 `class_id`：

- 安全域：`circuit_open`、`circuit_short`、`safety_sensor_invalid`、`safety_normal`。
- 低压域：`battery_depleted`、`charging_failure`、`terminal_resistance`、`voltage_normal`。

样例由本地 `synthetic.make_request` 即时生成，每次返回新的 `request_id`。类别仅用于选择模拟场景；样例没有把真实标签写入分类输入。未提供或未知类别不会替换为成功示例。

`DiagnosisRequest` 的主要字段：

```json
{
  "request_id": "manual-event-001",
  "device_id": "demo-device-001",
  "entry": "user",
  "complaint": "安全联锁回路报警，需要检查",
  "domain": null,
  "fault_id": null,
  "fault_time_ms": 10000,
  "samples": [],
  "latitude": 31.0,
  "longitude": 121.0
}
```

该最小请求缺失有效采样，只会得到未知状态；要运行正常感知/特征阶段，请加载授权样例并保留完整采样窗口。`entry=user` 必须提供非空主诉；`entry=telemetry` 必须绑定 `domain`（`safety` / `low_voltage`）和已配置 `fault_id`。推荐主动告警走 `/api/telemetry`，由服务绑定故障标识。

采样为 `{"timestamp_ms":10000,"values":{"resistance_ohm":120.0}}`。最多1000点，每点最多32字段。缺失值可显式为 `null`；NaN、Infinity、超出有限范围的数值不允许。坐标限制纬度 [-90,90]、经度 [-180,180]。领域所需字段与采样覆盖由特征/服务层验证；缺失、冲突、未激活模型不会伪装成正常预测。

`request_id` 与身份共同作为幂等键：同内容重复提交返回原诊断，改内容却复用同 ID 返回409。页面加载新样例会换 ID；手改已有 JSON 后应手动换 ID。

### 结果阅读

`DiagnosisResult` 包括：

- `outcome`：`unknown`、`review_required` 或 `resolved`；先读 `reason` 理解门控决定。
- `prediction`：可能为 `null`。有值时含 `class_id`、`scores`（**模型预测概率而非物理置信度**）、`accepted`、`reliability_reason`、模型版本。
- `evidence`：每项含 `condition`、`description`、`observed` 与 `status`（`supported` / `refuted` / `missing`）。`condition=root` 是按知识图 AND/OR 关系计算的聚合结论，不应对所有叶项简单取 AND；例如 OR 分支之一被反驳不代表聚合结论必定反驳。缺失不可等同“无故障”。
- `stages`：实际记录的 `perception`、`routing`、`execution`、`gating`、`feedback_pending`；被路由阻断时可能不进入执行。
- `attribution`：只有满足归因条件才存在；`safety_critical` 提示人工复核。
- `dispatch_candidates`：合成技能、库存、容量匹配及距离评分候选。距离为球面距离，不是道路里程；候选不是预约。
- `knowledge_version`、`model_version`、各类 ID 和时间：用于追溯。

### 遥测

```json
{
  "event_id": "event-001",
  "device_id": "demo-device-001",
  "domain": "safety",
  "samples": [
    {"timestamp_ms": 10000, "values": {"warning_active": 0.0}}
  ],
  "latitude": 31.0,
  "longitude": 121.0
}
```

上述安静事件不会触发诊断。服务按新增采样逐点检查持续告警、时间水位和冷却；返回例如 `{"status":"ignored","reason":"not_sustained","diagnosis":null,"diagnoses":[],"trigger_count":0}`。触发时 `status=triggered`、`reason=sustained_alarm`，`diagnoses`包含全部实际诊断、`trigger_count`为数量，兼容字段`diagnosis`只返回首条；其诊断仍可能是未知。单批跨多个60秒冷却期可生成多条诊断，批末恢复不会抹去前段触发。重复 `event_id` 同内容幂等，不同内容409。页面“转为遥测草稿”不会强行把告警改为真。

### 反馈、派单、模型激活

反馈示例：

```json
{
  "feedback_id": "feedback-001",
  "correct_class": "circuit_open",
  "actual_action": "专家复核合成窗口中的回路电阻",
  "repair_confirmed": false,
  "note": "仅合成演示，未进行真实维修"
}
```

派单提交 `service_id`（从当前`dispatch_candidates`取得，不使用虚构站点ID）与`action_id`（例如`assignment-001`）。必须已有有效归因，且服务站仍满足当前技能、库存、容量。真正占用由事务完成；响应 `status=synthetic_assignment_confirmed` 仍含 `repair_completed=false`。关闭派单后重放最初action_id仍返回冻结的原派单响应，不代表当前状态又变回已确认。

模型激活请求包含 `version`（使用训练输出的真实 `sim-lgbm-...` 版本）与 `action_id`（例如 `activate-001`）。版本须先由本地 CLI 注册，模型文件须校验通过。服务再验证合成测试宏 F1 门限和新旧同一冻结评测集回归要求；不存在或失败会返回业务错误，不自动生成模型。网页不上传模型文件。

上述操作 ID 用于服务端幂等/冲突检查。页面保留当前操作 ID 以便重试；修改操作内容时应使用新 ID，不能修改已占用的确认记录。

### 知识草稿与批准

1. `GET /api/knowledge` 获取当前裸 snapshot。
2. 修改 `version` 为未用版本，格式为 `knowledge-v` 后接正整数（例如 `knowledge-v2`），并按快照结构修改知识；不能覆盖已有版本。
3. 专家或管理员将**整个裸快照** POST 到 `/api/knowledge/proposals`，不是 `{"snapshot": ...}` 包装对象。服务进行结构与业务校验，成功仅成为草稿。
4. **不同身份的管理员**对 `/api/knowledge/{version}/publish` POST 无请求体。服务重新校验后发布并切换活动知识版本。

即使同一人是管理员，也不能批准自己的提案。读取当前快照不会返回未发布草稿；页面保存的草稿文本不是持久审批状态，刷新前请完成必要提交。

## 错误与请求体限制

所有 `DomainError` 使用其业务 HTTP 状态并统一返回：

```json
{"error":{"code":"forbidden","message":"当前角色没有执行该动作的权限"}}
```

典型状态：401 未认证（带 `WWW-Authenticate: Bearer`）；403 角色/独立审批不满足；404 不存在或不在授权范围；409 幂等冲突、资源条件或版本门限不满足；503 当前知识不可用。模型不可用在诊断流程内可能被受控捕获并返回 **200 + unknown**，应检查 `outcome` 与 `reason`，而非只看 HTTP 状态。

JSON 语法或请求模型错误统一为标准422形态，不回显原始输入：

```json
{"detail":[{"type":"json_invalid","loc":["body"],"msg":"请求体必须是合法JSON，且所有数值必须有限"}]}
```

客户端应发送 `Content-Type: application/json`。JSON NaN/Infinity、溢出数字及模型层非法字符串数值都返回422，而不是服务器序列化500。业务知识验证自身的 `DomainError` 仍使用 `error` 形态。

全请求体上限 **1 MiB（1,048,576字节）**。既检查声明长度，也累计实际流字节，缺失/伪造 Content-Length 不能绕过；超限413 `body_too_large`。建议在反向代理层另设连接、时间与速率限制；本地工作台不代表互联网生产部署。

## 验证范围

`tests/test_api.py` 使用临时目录中的真实 `Store`、`DiagnosisService`，覆盖所有业务端点认证、跨操作员隔离、角色伪造、非有限数和JSON错误、无活动模型的真实持久化诊断、幂等事务、遥测、反馈审计、缺失归因拒绝派单、知识独立审批、静态资源与 OpenAPI、流式请求限长及导入无运行目录副作用。没有用 mock 成功结果替代业务集成；活动模型训练与完整模型质量测试由对应模块测试负责。
