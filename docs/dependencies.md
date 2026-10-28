# 第三方依赖来源与许可证记录

本文件记录本次Python3.12本地验收环境实际安装的公开依赖，不给本仓库原创代码添加开源许可证，也不是完整的法律合规意见。

## 来源与锁定

- 应用/测试依赖声明：`pyproject.toml`。
- 固定解析版本、下载地址与哈希：`uv.lock`，源为公开PyPI（`pypi.org`、`files.pythonhosted.org`）。
- 许可证信息来自已安装发行包的`METADATA`及其附带的LICENSE文件；不是从内部文档复制。
- 未把第三方包源码、二进制库、模型权重打进Git仓库或应用wheel；运行时由包管理器单独安装。
- 不同平台的传递依赖可能不同，应以目标环境实际安装与发行包附带声明为准。

## 本次安装清单

| 发行包 | 版本 | 声明/核对到的许可证 |
|---|---|---|
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| certifi | 2026.10.23 | MPL-2.0 |
| click | 8.5.0 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| fastapi | 0.162.0 | MIT |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| iniconfig | 2.3.1 | MIT |
| Jinja2 | 3.1.6 | BSD-3-Clause（附带LICENSE核对） |
| joblib | 1.6.0 | BSD-3-Clause |
| lightgbm | 4.7.0 | MIT |
| markdown-it-py | 4.2.0 | MIT |
| MarkupSafe | 3.0.4 | BSD-3-Clause |
| mdurl | 0.1.2 | MIT |
| narwhals | 2.28.0 | MIT |
| networkx | 3.7 | BSD-3-Clause |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0（元数据表达式） |
| opentelemetry-api | 1.45.1 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 2.3.3 | BSD-3-Clause |
| pluggy | 1.7.0 | MIT |
| pydantic | 2.14.1 | MIT |
| pydantic_core | 2.50.1 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pytest | 8.4.2 | MIT |
| python-dateutil | 3.0.0 | 元数据标注Dual License；附带文件按贡献时期区分Apache-2.0与历史BSD条款，参见原包LICENSE |
| python-multipart | 0.0.32 | Apache-2.0 |
| pytz | 2026.5 | MIT |
| rich | 15.0.0 | MIT |
| ruff | 0.17.3 | MIT |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| scipy | 1.18.1 | 主项目BSD-3-Clause；二进制内组件须另看发行包完整LICENSE |
| shellingham | 1.5.4 | ISC |
| six | 1.17.0 | MIT |
| starlette | 1.10.0 | BSD-3-Clause |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| typer | 0.27.3 | MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| tzdata | 2026.5 | Apache-2.0（发行包元数据；时区数据可能另有声明） |
| uvicorn | 0.54.0 | BSD-3-Clause |

本表记录原包声明，不替代保留上游完整版权、NOTICE、数据或捆绑库许可证。若未来分发容器、安装器、离线依赖包或数值库二进制，应另做完整SBOM及第三方通知核对；当前仅上传原创工程代码与锁文件。

## 构建/宿主工具边界

`uv`、Hatchling构建后端、Python、macOS OpenMP运行库、Node及用于浏览器烟测的Chrome不随本仓库分发。本表没有穷举隔离构建环境、操作系统组件或浏览器的许可证；安装/分发这些工具应遵循其各自条款。仓库没有模型下载器，也没有第三方生成式模型权重需要许可分发。
