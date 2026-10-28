# 输入与输出数据字典

全部示例是本仓库的合成协议；单位不是对任何现实设备通信协议的确认。

## 请求

| 字段 | 类型/限制 | 含义 |
|---|---|---|
| request_id | 1–100字符ASCII标识 | 同一身份内幂等键；内容变化须换ID |
| device_id | 同上 | 合成设备；训练分组标识，不是VIN |
| entry | user / telemetry | 用户或主动入口 |
| complaint | 最长2000字符 | 用户入口必填；用于领域分类，不直接确定根因 |
| domain | safety / low_voltage / null | 可指定领域，用户文本推断若冲突则拒识 |
| fault_id | 标识或null | 领域故障定义，主动入口必填 |
| fault_time_ms | 整数，0至2^63−1 | 事件时间轴毫秒，窗口中心 |
| samples | 最多1000点 | timestamp_ms（同上范围）+ 数值字段字典 |
| latitude/longitude | 合法经纬度 | 只给合成服务站距离计算使用 |

所有数据模型拒绝额外字段和NaN/Infinity。一个采样点最多32个信号；`null`代表真实缺失，不能当0。

## 信号

| 领域 | 字段 | 本仓库单位/取值 |
|---|---|---|
| safety | resistance_ohm | 欧姆 |
| safety | supply_voltage | 伏 |
| low_voltage | voltage | 伏 |
| low_voltage | current_a | 安，允许正负方向 |
| low_voltage | engine_running | 0/1 |
| low_voltage | terminal_drop_v | 伏 |
| 两域 | warning_active | 0/1 |
| 两域 | dtc | 非负整数，合成告警编码 |

## 数值特征

定义与顺序唯一来源是`features.FEATURE_COLUMNS`，制品在加载时验证一致性。

**safety**：电阻中位数/min/max/std；供电电压min/mean；warning最大值；DTC激活比例。

**low_voltage**：电压min/mean/std/每秒线性斜率；电流mean/min；engine_running最大值；端子压降max/mean；warning最大值；DTC激活比例。

均只取窗口内点；std使用总体窗口标准差，DTC只使用是否激活比例而非纯标签查表。出现缺失、冲突采样或非有限统计量时，主服务不会调用分类确认流程。

## 诊断响应

- `diagnosis_id`：持久化身份；独立于客户端request_id。
- `outcome`：unknown / review_required / resolved。
- `reason`：本次停止或确认原因。
- `prediction`：模型类别、各类实际分数、模型版本、拒识状态；可能为null。
- `evidence`：可执行条件的supported/refuted/missing；不是模型编写的解释。
- `attribution`：类别映射出的业务归因及独立solution_id；缺乏证据时为null。
- `knowledge_version` / `model_version`：本次冻结版本；后续发布不追改旧结果。
- `dispatch_candidates`：当前快照下的合成资源候选，不保证稍后仍有容量。
- `stages`：实际经过的固定阶段。
- `created_at`：记录时间，UTC。

## 专家反馈

`feedback_id`是幂等键；`correct_class`是领域内标签；`actual_action`记录实际人工输入；`repair_confirmed`是独立字段，不修改原诊断outcome。标签确认不意味着已修复。导出训练记录保留原始请求、设备、诊断事件ID、标签和反馈ID。

## 两种“置信度”不要混用

- 模型`scores`：LightGBM多分类输出，未作严格概率校准。
- 模式挖掘`confidence`：共现条件支持比值，属于关联统计，不是分类准确率，更不是因果置信度。

## API与存储限制

API请求体上限1MiB，逐块验证实际大小；CLI诊断输入文件上限2MB。模型权重、SQLite、令牌和完整运行制品都在Git忽略目录，发布检查另外扫描Git可见文件、暂存区和历史。
