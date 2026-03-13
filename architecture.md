# Factor Investing System Architecture (AlphaVibe)

## 1. 系统概述 (System Overview)
本项目旨在构建一个端到端的因子投资（Factor Investing）研究与回测框架。系统设计遵循“高内聚、低耦合”的原则，数据流采用单向 Pipeline 模式：**数据获取 -> 清洗存储 -> 因子计算 -> 截面/时序分析 -> 组合优化 -> 回测归因**。

## 2. 目录结构 (Directory Structure)
```text
alpha_vibe/
├── data/                   # 本地数据存储 (Parquet为主)
│   ├── raw/                # 原始行情与基本面数据
│   └── processed/          # 清洗对齐后的标准格式数据
├── core/                   # 核心基础组件
│   ├── data_fetcher/       # 异步数据抓取模块
│   └── database/           # 数据库连接与ORM抽象
├── factors/                # 因子库
│   ├── base.py             # 因子基类 (定义计算规范)
│   ├── momentum/           # 动量类因子实现
│   ├── value/              # 估值类因子实现
│   └── alternative/        # 另类数据因子 (基于LLM提取等)
├── backtest/               # 回测引擎
│   ├── engine.py           # 向量化回测核心逻辑
│   ├── performance.py      # 绩效评估 (Sharpe, MaxDrawdown)
│   └── portfolio.py        # 投资组合构建与权重优化
├── notebooks/              # Jupyter研究环境 (用于探索性数据分析)
├── scripts/                # 自动化任务脚本 (Cron Jobs)
├── tests/                  # 单元测试
├── config.yaml             # 全局配置文件 (API Keys, 路径等)
├── requirements.txt        # 依赖包
└── architecture.md         # 架构文档
3. 技术栈选型 (Technology Stack)
3.1 核心计算与语言
语言: Python 3.10+

矩阵运算: numpy, pandas (重度依赖列式计算，避免使用 iterrows 或原生的 for 循环)。

并发处理: multiprocessing 用于多核并行计算横截面因子，突破 CPU 瓶颈。

3.2 数据获取与网络请求
核心数据源: EODHD API (覆盖全球历史行情与基本面)。

网络 IO: 使用 asyncio 配合异步 HTTP 客户端（或将 requests 结合线程池）进行高并发的 API 批量拉取，确保每日盘后数据快速入库。注意：API Client 应在单个请求或独立任务内部初始化，避免全局共用引发线程安全问题。

3.3 数据存储与管理
本地高性能存储: 行情和因子值统一序列化为 .parquet 格式，大幅提升 I/O 速度与 pandas 的兼容性。

大规模数据处理: 针对海量 Tick 数据或底层明细数据的离线聚合分析，通过 Hive SQL 进行 ETL，再将结果落盘。

元数据与关系型存储: 股票代码映射、交易日历等结构化数据可轻量级存入 SQLite，或在服务器端使用 PostgreSQL/MySQL。

3.4 回测与评价工具
单因子评价: 基于开源的 Alphalens 计算 IC/IR、换手率及进行分层回测。

组合优化: 结合 scipy.optimize 或开源量化框架（如 Qlib 的部分模块）进行带约束的权重求解。

3.5 另类数据与模型推理（扩展层）
非结构化数据提纯: 利用本地部署的 vLLM 引擎（如挂载 Qwen 等视觉/语言模型）或 Azure OpenAI 服务，批量提取研报情绪或财务报表特征，降维成结构化 Alpha 因子。

3.6 任务调度与运维
操作系统环境: Linux 环境。

自动化调度: 采用 Shell 脚本 (.sh) 结合系统自带的 cron 定时任务，实现每日盘后自动化 Pipeline（拉取数据 -> 跑批计算 -> 生成明日调仓信号）。