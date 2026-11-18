# -*- coding: utf-8 -*-
"""
翻车复现（P12 指令产物）：换元计数据重跑 k-均值——未标准化，当场崩
预期读数（真实运行锁定）：低收入簇 141 / 中产 111 / 富裕 48
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---- 1. 数据初检 ----
df = pd.read_csv("data/mall_customers_yuan.csv")
print(f"行列数: {df.shape}")
print("各列缺失值:\n", df.isnull().sum().to_string())
print("数值列统计:\n", df[["age", "income", "balance", "product_count"]].describe().round(2).to_string())

# ---- 2. 重跑聚类（k=3、random_state=42，其他不动——故意未标准化）----
X = df[["income", "balance"]]
km = KMeans(n_clusters=3, random_state=42)
df["cluster"] = km.fit_predict(X)

counts = df["cluster"].value_counts().sort_index()
print("\n各簇人数:")
for c, n in counts.items():
    stat = df[df["cluster"] == c][["income", "balance"]].mean()
    print(f"  簇 {c}: {n} 人 | 平均收入 {stat['income']:.1f} 万 | 平均余额 {stat['balance']:,.0f} 元")

# ---- 3. 重画聚类图 ----
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
colors = ["#4C72B0", "#DD8452", "#55A868"]
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["income"], sub["balance"], s=28, c=colors[c], alpha=0.75,
               label=f"簇 {c}（{len(sub)} 人）")
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           marker="*", s=520, c="#B03A2E", edgecolors="white", linewidths=1.2,
           zorder=5, label="簇中心")
ax.set_xlabel("年收入（万元）")
ax.set_ylabel("账户平均余额（元）")
ax.set_title("重跑结果：余额以元计、未标准化——低收入簇膨胀至 141（翻车现场）")
ax.legend()
ax.yaxis.set_major_formatter(lambda v, _: f"{v/10000:.0f}万")
fig.tight_layout()
fig.savefig("outputs/02_yuan_fail.png")
print("\n图已保存: outputs/02_yuan_fail.png")

# ---- 4. 更新分群报告 ----
report_lines = ["# 分群报告（元计数据重跑——翻车版）", "",
                "| 簇 | 人数 | 平均年龄 | 平均收入(万) | 平均余额(元) | 平均产品数 |",
                "|---|---|---|---|---|---|"]
for c in sorted(df["cluster"].unique()):
    g = df[df["cluster"] == c]
    report_lines.append(
        f"| {c} | {len(g)} | {g['age'].mean():.0f} | {g['income'].mean():.1f} | "
        f"{g['balance'].mean():,.0f} | {g['product_count'].mean():.1f} |")
report_lines += ["", "对照上一轮 125/125/50：低收入簇吸走 16 个中产——量纲陷阱现场。"]
with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print("报告已更新: generated/mall_report.md")
