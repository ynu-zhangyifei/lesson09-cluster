# -*- coding: utf-8 -*-
"""
第 2-4 步（修正版）：k-均值聚类 + 聚类结果可视化 + 分群报告
（由修正指令 A 生成：加标准化 + n_init=10 + 修正前后对比图）
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 中文显示设置
matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/mall_customers_yuan.csv")

# ---------- 第 2 步（修正版）：标准化 + k-均值聚类 ----------
X_raw = df[["income", "balance"]]
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)          # 标准化：量纲拉齐，两特征等权
km = KMeans(n_clusters=3, random_state=42, n_init=10)   # 显式 n_init=10
labels = km.fit_predict(X)
df["cluster"] = labels

# ---------- 第 3 步（修正版）：原始坐标 + 标准化坐标并排对比图 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
colors = ["#DD8452", "#4C72B0", "#55A868"]

# 左图：原始坐标
ax = axes[0]
for c in range(3):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["income"], sub["balance"] / 10000, s=40, alpha=0.7,
               color=colors[c], label=f"簇 {c}（{len(sub)} 人）")
centers_raw = scaler.inverse_transform(km.cluster_centers_)  # 簇中心逆变换回原始坐标
ax.scatter(centers_raw[:, 0], centers_raw[:, 1] / 10000,
           marker="*", s=600, c="red", edgecolors="black", linewidths=1.2,
           zorder=5, label="簇中心")
ax.set_title("修正后 · 原始坐标（余额单位：万元）", fontsize=13)
ax.set_xlabel("年收入 income（万元）", fontsize=11)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# 右图：标准化坐标
ax = axes[1]
for c in range(3):
    sub = df[df["cluster"] == c]
    ax.scatter(X[df["cluster"].to_numpy() == c, 0],
               X[df["cluster"].to_numpy() == c, 1],
               s=40, alpha=0.7, color=colors[c], label=f"簇 {c}（{len(sub)} 人）")
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           marker="*", s=600, c="red", edgecolors="black", linewidths=1.2,
           zorder=5, label="簇中心")
ax.set_title("修正后 · 标准化坐标（无单位，均值0方差1）", fontsize=13)
ax.set_xlabel("income（标准化后）", fontsize=11)
ax.set_ylabel("balance（标准化后）", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

fig.suptitle("标准化不是改变客户，是改变度量衡：两特征等权参与距离", fontsize=15, y=0.99)
plt.tight_layout()
plt.savefig("outputs/04_compare_scale.png", dpi=150)
print("已保存 outputs/04_compare_scale.png")

# ---------- 第 4 步：分群报告（标准化版） ----------
summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入万=("income", "mean"),
    平均余额万=("balance", lambda s: s.mean() / 10000),
    平均持有产品数=("product_count", "mean"),
).round(2)
print("\n", summary.to_string())

order = summary["平均收入万"].sort_values().index.tolist()
biz_names = {}
name_pool = ["低收低余的年轻客群", "中收中余的主力客群", "高收高余的富裕客群"]
for cl, nm in zip(order, name_pool):
    biz_names[cl] = nm

lines = ["# 客户分群分析报告（k-均值，k=3，标准化后）", "",
         "## 各簇统计概览", "", summary.to_string(), "",
         "## 客群业务命名", ""]
for cl in order:
    row = summary.loc[cl]
    lines.append(
        f"- **簇 {cl} · {biz_names[cl]}**：共 {int(row['人数'])} 人"
        f"（占 {row['人数']/300*100:.1f}%），平均年龄 {row['平均年龄']} 岁、"
        f"平均年收入 {row['平均收入万']} 万元、平均余额 {row['平均余额万']} 万元、"
        f"平均持有 {row['平均持有产品数']} 个产品。"
    )
lines += ["", "## 一句话结论", "",
         "300 位客户在「收入-余额」平面上自然分成三块："
         "**低收低余的年轻客群**（约四成，潜力客户，适合开户与首卡渗透）、"
         "**中收中余的主力客群**（约四成，银行基本盘，适合交叉销售理财与基金）、"
         "**高收高余的富裕客群**（约两成，高价值客户，适合专属客户经理与私行服务）。"]

with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("已保存 generated/mall_report.md（标准化版）")
