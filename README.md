# 智能诊断数字技师

**固定诊断流程 + 真实 LightGBM + 可执行知识图 + 本地工作台。**

从零实现的独立参考工程，包含实际训练、API、网页、CLI、SQLite和自动化测试。所有设备、波形、标签、服务站、阈值均为原创合成内容；不包含任何企业生产源码、内部文档、真实设备数据或生产模型，合成评测不代表现实故障诊断性能。

这不是自由ReAct Agent，也不需要生成式大模型或云端API Key。它处理两个领域：安全电路与低压供电。

## 能做什么

- **双入口**：用户主诉经TF-IDF/逻辑回归路由；遥测逐点检查持续告警，包含水位、跨批缓存、重启恢复、幂等及冷却。
- **真实学习**：原创电气模拟数据，设备级train/validation/test分组；分域LightGBM、多组有限超参、先验/线性基线、验证集拒识阈值和冻结评测报告。
- **证据确认**：数值窗口进入模型，`class_id → attribution_id → solution_id`连接知识对象；networkx条件图执行AND/OR三值验证，不把分数当根因证明。
- **失败有明确结果**：缺信号、缺模型、数值异常、范围外输入、主诉冲突不能自动确认；安全关键结果必须人工复核。
- **可追踪操作**：原诊断冻结，专家反馈独立记录；知识草稿需要独立审批；模型注册、非退化门控、激活与历史批准版本回滚。
- **合成派单**：技能/库存/工位筛选与距离排序，事务内预留，取消归还资源；无真实预约、维修或设备控制。
- **可操作工作台**：诊断、遥测、证据、记录、反馈、派单/关闭、知识审批、模型激活/回滚、审计。

```text
主诉 / 持续遥测
  → 感知 → 路由 → 数值特征 + LightGBM + 知识条件
  → 可靠性 / 证据 / 安全门
  → unknown / review_required / resolved
  → 人工处置与反馈 → 候选训练或知识更新 → 审批
```

`resolved`仅表示满足本合成协议的诊断条件，**不是维修完成**；`review_required`也不是专家已经批准。

## 1. 安装

需要Python **3.12**和[uv](https://docs.astral.sh/uv/)。在仓库根目录执行：

```bash
uv sync --locked --python 3.12
uv run diag --help
```

Apple Silicon上的LightGBM还需要OpenMP运行库。如果加载时报缺少`libomp`，安装：

```bash
HOMEBREW_NO_INSTALL_CLEANUP=1 brew install libomp
```

本地验收环境为Apple Silicon/macOS、Python3.12；其他系统尚未实机验收。运行不需要Docker、GPU或Java。首次安装会从公开软件源下载依赖，安装完成后诊断、训练与网页不调用外部服务。

## 2. 一条命令跑真实演示

```bash
uv run diag demo --directory runtime/demo
```

这会在独立目录内生成240台合成设备的训练数据，训练两个模型、注册并激活，执行八类诊断、主动告警去重和专家反馈。报告位于`runtime/demo/demo-report.json`。

- 不是预录响应，也没有无模型时的“成功”替代路径。
- 演示身份令牌不输出，数据与模型不进入Git。
- 目录必须为空或不存在；重跑用`runtime/demo-2`，不能覆盖原结果。
- 演示目录与下面的交互工作台默认目录独立，不会自动替工作台激活模型。

## 3. 启动交互工作台

以下命令在同一仓库根目录执行。先创建本地身份，每个令牌只显示一次，请自己保管，不要粘贴到聊天、源码或截图中：

```bash
uv run diag user local-admin --role admin
uv run diag user local-expert --role expert
uv run diag user local-operator --role operator
```

重复身份不会被覆盖。`diag user`是本机文件系统管理员命令，不是匿名HTTP注册接口。

将管理员令牌临时读入当前终端环境，输入不会回显或写入命令历史：

```bash
export DIAG_TOKEN="$(uv run python -c 'import getpass; print(getpass.getpass("管理员令牌: "))')"
mkdir -p runtime
uv run diag train --devices 240 --seed 17 > runtime/train-report.json
VERSION="$(uv run python -c 'import json; print(json.load(open("runtime/train-report.json"))["version"])')"
uv run diag activate "$VERSION"
unset DIAG_TOKEN
uv run diag serve --port 8000
```

浏览器打开 **http://127.0.0.1:8000**：

1. 用operator令牌登录，选择合成场景，加载样例，提交诊断。
2. 查看阶段、真实模型分数、聚合根证据及解释分支、归因与方案。
3. 由样例准备遥测，核对后提交；正常类不强行改成告警。
4. 换expert登录，可复核反馈、确认合成派单、取消或完成。
5. 知识提案由expert提交，再由不同身份的admin发布；admin还可激活/回滚模型。

网页令牌仅存在页面内存中，刷新需重新登录。不要把服务直接暴露到互联网。

## 4. CLI与反馈训练

需要授权的CLI命令读取`DIAG_TOKEN`。用上面的隐藏输入方式按需设置相应身份令牌。

```bash
uv run diag example --class-id battery_depleted --seed 12 > runtime/request.json
uv run diag diagnose runtime/request.json
uv run diag export-feedback > runtime/feedback.json
```

管理员执行真实反馈回流：

```bash
uv run diag train --devices 240 --seed 17 --include-feedback > runtime/candidate-report.json
```

必须保持相同seed与设备数，才能与旧模型使用同一冻结评测分区。只有经专家确认的记录参与导出；验证/测试设备、重复或不合法窗口不会加入训练，接收/排除统计写入报告。冲突专家标签会阻断导出，当前没有应用内标签仲裁界面。

从候选报告取得真实`version`后执行`diag activate VERSION`。候选模型不能低于当前模型的分域宏F1；未通过门控不生效。回滚命令：

```bash
uv run diag rollback PREVIOUS_APPROVED_VERSION "说明本次回滚原因"
```

这里的大写版本是占位符，必须替换为本机历史批准的`sim-lgbm-...`版本。激活、回滚不会重写过去的诊断。

`diag mine INPUT_FILE`从带专家标签的事件记录计算支持度、置信度和lift，只产生候选关联，不自动发布知识。记录格式为`event_id`、`device_id`、`features`、`confirmed_class`，具体示例见`tests/test_mining.py`。

## 5. 测试与打包

```bash
uv run pytest -q
uv run ruff check src tests scripts
node --check src/intelligent_diagnosis/static/app.js
uv build
uv run python scripts/check_release.py
```

应用运行不需要Node。开发验收使用Node检查JS语法，并可执行真实浏览器烟测：

```bash
node scripts/browser_smoke.mjs
uv run python scripts/check_wheel.py dist/intelligent_diagnosis_system-0.1.0-py3-none-any.whl
```

浏览器脚本要求macOS上已安装Google Chrome、Node20以上及本项目`.venv`依赖；使用新建临时浏览器配置和合成运行目录，不读取用户浏览器配置，不下载浏览器，不伪造业务响应，结束时清理临时服务与目录。覆盖范围与限制见[实际验收记录](docs/validation.md)。

安装包检查默认离线，新机器缓存不足时需明确添加`--allow-download`从公开PyPI下载依赖。wheel包含网页模板与静态资源；可在独立环境安装`dist/*.whl`后运行`diag`，不需要源码目录。

测试覆盖真实训练、窗口与拒识、图结构和三值逻辑、权限/幂等/事务、分批不变遥测、多触发响应、专家回流、版本门控及HTTP边界。发布扫描检查工作树、暂存区与历史中的禁止路径/常见秘密特征，不等于完整秘密检测，也不替代人工审核。

## 目录

```text
src/intelligent_diagnosis/
  schemas.py / features.py / synthetic.py   类型、特征、原创波形
  routing.py / ml.py                        文本路由与真实模型训练/推理
  knowledge.py / mining.py / dispatch.py    知识、关联挖掘、资源排序
  storage.py / service.py                   持久化与确定性业务流程
  api.py / cli.py / templates/ / static/    API、命令行与工作台
tests/                                     自动化回归
docs/                                      架构、接口、数据字典与验收
scripts/check_release.py                   发布内容检查
scripts/check_wheel.py                     干净环境安装验收
scripts/browser_smoke.mjs                  真实Chrome表单验收
uv.lock                                    依赖锁定
```

## 配置与安全边界

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| DIAG_RUNTIME_DIR | runtime | 本地SQLite及内容寻址制品 |
| DIAG_MODEL_DIR | models | 模型文本及元数据 |
| DIAG_TOKEN | 无 | 需要授权的CLI命令身份 |

`.env.example`仅作说明，程序**不自动加载.env文件**。CLI/API必须指向同一运行与模型目录；相对目录由启动时工作目录解释。不要提交runtime、模型权重、令牌、真实数据或日志；也不要关闭`.gitignore`来上传运行制品。

这是单机、单工作台参考实现，不是企业多租户系统。SQLite写事务内会执行诊断与发布检查，吞吐能力有限；没有生产并发SLA。哈希保障意外损坏可发现，不是抵御本机管理员的数字签名。模型分数未严格校准，数值范围拒识不能覆盖全部未知故障，知识回归不能证明现实因果。

现实接入还需要数据授权、真实协议与单位核验、时间同步、设备留出评测、模型校准、专业安全审核、SSO/租户隔离、备份恢复和现场验收。没有ONNX运行时、生产接口、云部署或真实维修执行；不把这些写成已实现。

## 文档

- [架构与工程边界](docs/architecture.md)
- [输入输出数据字典](docs/data-dictionary.md)
- [API与认证](docs/api.md)
- [实际验收记录](docs/validation.md)
- [依赖来源与许可证记录](docs/dependencies.md)

本仓库未附加开源许可证，不授予未明确约定的开源使用许可。
