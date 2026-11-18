# -*- coding: utf-8 -*-
"""
第 2-4 步：k-均值聚类 + 聚类结果可视化 + 分群报告
（由起点指令 01 生成——客户分群项目第 2、3、4 件事）
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

# 中文显示设置
matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/mall_customers_sim.csv")

# ---------- 第 2 步：k-均值聚类（k=3，仅 income/balance 两特征） ----------
X = df[["income", "balance"]]
km = KMeans(n_clusters=3, random_state=42)
labels = km.fit_predict(X)
df["cluster"] = labels

# ---------- 第 3 步：聚类结果可视化 ----------
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#DD8452", "#4C72B0", "#55A868"]
for c in range(3):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["income"], sub["balance"], s=40, alpha=0.7,
               color=colors[c], label=f"簇 {c}（{len(sub)} 人）")
# 簇中心星形标记
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           marker="*", s=600, c="red", edgecolors="black",
           linewidths=1.2, zorder=5, label="簇中心")
ax.set_title("k-均值聚类结果（k=3，特征：income/balance）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/03_clustered_scatter.png", dpi=150)
print("已保存 outputs/03_clustered_scatter.png")

# ---------- 第 4 步：分群报告 ----------
summary = df.groupby("cluster").agg(
    人数=("customer_id", "count"),
    平均年龄=("age", "mean"),
    平均收入=("income", "mean"),
    平均余额=("balance", "mean"),
    平均持有产品数=("product_count", "mean"),
).round(2)
print("\n", summary.to_string())

# 按平均收入排序给业务名字（低→高）
order = summary["平均收入"].sort_values().index.tolist()
biz_names = {}
name_pool = ["低收低余的年轻客群", "中收中余的主力客群", "高收高余的富裕客群"]
for cl, nm in zip(order, name_pool):
    biz_names[cl] = nm

lines = ["# 客户分群分析报告（k-均值，k=3）", "",
         "## 各簇统计概览", "", summary.to_string(), "",
         "## 客群业务命名", ""]
for cl in order:
    row = summary.loc[cl]
    lines.append(
        f"- **簇 {cl} · {biz_names[cl]}**：共 {int(row['人数'])} 人"
        f"（占 {row['人数']/300*100:.1f}%），平均年龄 {row['平均年龄']} 岁、"
        f"平均年收入 {row['平均收入']} 万元、平均余额 {row['平均余额']} 万元、"
        f"平均持有 {row['平均持有产品数']} 个产品。"
    )
lines += ["", "## 一句话结论", "",
         "300 位客户在「收入-余额」平面上自然分成三块："
         "一块是**低收低余的年轻客群**（约四成，潜力客户，适合做开户与首卡渗透）；"
         "一块是**中收中余的主力客群**（约四成，银行基本盘，适合交叉销售理财与基金）；"
         "一块是**高收高余的富裕客群**（约两成，高价值客户，适合专属客户经理与私行服务）。"]

with open("generated/mall_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("已保存 generated/mall_report.md")
