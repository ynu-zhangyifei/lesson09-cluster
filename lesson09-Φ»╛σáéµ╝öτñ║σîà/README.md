# lesson09 课堂演示包（自包含）

> 用途：第 9 次课《聚类与客户分群》当堂演示。本目录自包含——`cd` 进来即可逐步运行，
> 不依赖课程仓。全部脚本已实跑验证（2026-11-13 00:08），读数与幻灯片逐字一致。
> 环境：`conda activate base`（或 `/opt/homebrew/Caskroom/miniconda/base/bin/python3`）。
> **逐字稿**（含每节拍演示操作盒+代课上手卡+应急兜底卡）：
> `_md源文件/讲义/2026秋-第九次课-聚类与客户分群逐字稿.md`——代课老师拿稿即可上课。

## 课堂动线（对应幻灯片页码）

| 步 | 幻灯片 | 演示动作 | 命令 | 预期读数 |
|---|---|---|---|---|
| 1 | P10 | 贴起点指令全文（或先只贴初检任务） | `python3 generated/01_load_and_viz.py` | 300 行 7 列；图 01_raw_scatter.png 肉眼三块 |
| 2 | P11 | 起点跑通 | `python3 generated/02_起点版_kmeans.py` | **125 / 125 / 50**（低收/中产/富裕） |
| 3 | P12→P13 | 贴翻车复现指令 → 当场崩 | `python3 generated/02b_yuan_rerun.py` | **141 / 111 / 48**（低收入簇吸走 16 个中产） |
| 4 | P15→P16 | 贴修正指令A → 拉回 | `python3 generated/02_kmeans_cluster.py` | **126 / 124 / 50**；04_compare_scale.png 左右对照 |
| 5 | P18→P19 | 贴修正指令B → 选k双证据 | `python3 generated/03_choose_k.py` | 肘部 k=3；轮廓最高 k=2（0.7351），k=3 为 0.6902 |
| 6 | P21→P22 | 质心迭代 2×2 画布 | `python3 generated/04_iteration_demo.py` | 3 轮收敛；inertia 逐轮下降 |

## 目录

- `data/` — mall_customers_sim.csv（万元计·起点）、mall_customers_yuan.csv（元计·翻车）
- `generated/` — 6 份可运行脚本（02b 为翻车复现，即 P12 指令的预期产物）
- `outputs/` — 6 张分析图（运行即重新生成，已预跑一遍）
- `prompts/` — 7 份指令全文（与幻灯片 7 页全文页同源：P10/P12/P15/P18/P21/P26/P29）
- `webapp/` — 成品看板页（节拍 18 "草图变看板"演示物与兜底）
- `diagrams/` — Mermaid 图即代码三图（流程/用例/类图，节拍 19 展示用）

## 提醒

- 脚本一律在**本目录**下运行（内部用相对路径 `data/`、`outputs/`）
- 上课时把 prompts/ 里的指令**整段贴进 Trae CN** 让 AI 现场写代码——AI 写出来的与
  `generated/` 里的等价；万一现场翻车，直接跑现成脚本兜底
- 翻车读数 141/111/48 是真实运行结果，若课堂环境复现出小偏差，以当场数字为准讲原理
