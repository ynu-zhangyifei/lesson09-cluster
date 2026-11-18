# -*- coding: utf-8 -*-
"""
任务 2-4：k-均值聚类 + 聚类结果可视化 + 分群报告
特征只用 income（年收入）与 balance（账户平均余额），customer_id 不进入模型。
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.cluster import KMeans

# ---------- 中文字体设置 ----------
候选字体 = ["Songti SC", "STHeiti", "PingFang SC",
            "Microsoft YaHei", "SimHei", "Noto Sans SC"]
已装字体 = {f.name for f in fm.fontManager.ttflist}
可用字体 = [f for f in 候选字体 if f in 已装字体]
if not 可用字体:
    raise RuntimeError("未找到可用的中文字体，请检查系统字体安装")
matplotlib.rcParams["font.sans-serif"] = 可用字体
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 读入数据 ----------
df = pd.read_csv("data/mall_customers_sim.csv", encoding="utf-8-sig")

# 只取两个特征；customer_id 是编号，绝不能进模型
特征 = ["income", "balance"]
X = df[特征]

# ---------- 任务 2：k-均值聚类 ----------
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10)
簇编号 = km.fit_predict(X)
df["cluster"] = 簇编号  # 把簇编号写回数据框

print("=" * 55)
print("二、k-均值聚类（k=3，特征：income、balance）")
print("=" * 55)
print(f"参与聚类的特征：{特征}")
print(f"簇编号已写回数据框，列名：cluster")
print(f"各簇人数：{pd.Series(簇编号).value_counts().sort_index().to_dict()}")
print("\n簇中心（原始单位：万元）：")
中心表 = pd.DataFrame(km.cluster_centers_, columns=["平均年收入", "平均余额"])
中心表.index.name = "簇"
print(中心表.round(2).to_string())

# ---------- 任务 3：聚类结果可视化 ----------
颜色表 = {0: "#DD8452", 1: "#4C72B0", 2: "#55A868"}
fig, ax = plt.subplots(figsize=(10, 7))

for 簇 in range(k):
    子集 = df[df["cluster"] == 簇]
    ax.scatter(子集["income"], 子集["balance"],
               s=45, alpha=0.7, color=颜色表[簇],
               edgecolors="white", linewidths=0.5,
               label=f"簇 {簇}（{len(子集)} 人）")
    # 在簇中心旁边标注该簇人数
    ax.annotate(f"簇 {簇}\n{len(子集)} 人",
                xy=(km.cluster_centers_[簇, 0], km.cluster_centers_[簇, 1]),
                xytext=(14, 14), textcoords="offset points",
                fontsize=11, fontweight="bold", color=颜色表[簇],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75))

# 簇中心用大星形标记
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           marker="*", s=700, c="red", edgecolors="black",
           linewidths=1.2, zorder=5, label="簇中心")

ax.set_title("k-均值客户分群结果（k=3，按收入与余额划分）", fontsize=16)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/03_clustered_scatter.png", dpi=150)
print("\n已保存聚类散点图：outputs/03_clustered_scatter.png")

# ---------- 任务 4：分群报告 ----------
统计 = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入=("income", "mean"),
    平均余额=("balance", "mean"),
    平均产品数=("product_count", "mean"),
).round(2)
统计["占比"] = (统计["人数"] / len(df) * 100).round(1)

# 按平均收入从低到高排序，据此给出业务名字（名字与数据实际情况对应）
排序 = 统计.sort_values("平均收入").index.tolist()
名称池 = ["年轻起步客群", "稳健主力客群", "富裕尊享客群"]
业务名 = {簇: 名称池[i] for i, 簇 in enumerate(排序)}

# 给聚类结果补一列业务名字，方便业务方直接使用
df["客群"] = df["cluster"].map(业务名)

print("\n" + "=" * 55)
print("三、分群画像")
print("=" * 55)
画像 = 统计.copy()
画像.index = [f"簇 {c}（{业务名[c]}）" for c in 统计.index]
print(画像.to_string())

# 拼 markdown 报告
报告 = [
    "# 零售银行客户分群分析报告",
    "",
    "## 一、分析概要",
    "",
    f"- 数据：`data/mall_customers_sim.csv`，共 {len(df)} 位客户",
    "- 聚类特征：`income`（年收入，万元）、`balance`（账户平均余额，万元）；"
    "`customer_id` 仅作编号，未参与建模",
    f"- 方法：k-均值聚类（KMeans），簇数 k={k}，random_state=42",
    "",
    "## 二、各群规模与画像",
    "",
    "| 簇 | 业务客群 | 人数 | 占比 | 平均年龄(岁) | 平均年收入(万元) | 平均余额(万元) | 平均持有产品数 |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for 簇 in 统计.index:
    r = 统计.loc[簇]
    报告.append(
        f"| {簇} | {业务名[簇]} | {int(r['人数'])} | {r['占比']}% | "
        f"{r['平均年龄']} | {r['平均收入']} | {r['平均余额']} | {r['平均产品数']} |"
    )

报告 += ["", "## 三、分群解读", ""]

# 用簇的排序位置决定描述语气，保证与数据一致
描述模板 = {
    0: "收入与余额都处于最低区间，平均年龄最轻，账户资金沉淀薄，"
       "属于刚进入职场、以工资代发和日常消费为主的客户，价值提升空间大。",
    1: "收入与余额居中，是银行客户的基本盘，"
       "已有稳定资金沉淀，是理财、基金等交叉销售的主力人群。",
    2: "收入与余额显著高于其他两群，平均年龄偏大，"
       "资金实力强，适合专属客户经理跟进与财富管理服务。",
}
for i, 簇 in enumerate(排序):
    r = 统计.loc[簇]
    报告.append(
        f"- **簇 {簇} · {业务名[簇]}**：{int(r['人数'])} 人，占 {r['占比']}%；"
        f"平均年龄 {r['平均年龄']} 岁，平均年收入 {r['平均收入']} 万元，"
        f"平均余额 {r['平均余额']} 万元，平均持有 {r['平均产品数']} 个产品。"
        f"{描述模板[i]}"
    )

报告 += ["", "## 四、业务命名", ""]
for i, 簇 in enumerate(排序):
    报告.append(f"- **{业务名[簇]}**（簇 {簇}，{int(统计.loc[簇, '人数'])} 人）")

with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(报告) + "\n")

print("\n已保存分群报告：generated/mall_report.md")
print("\n业务名字：" + "；".join(f"簇 {c} → {业务名[c]}" for c in 排序))
