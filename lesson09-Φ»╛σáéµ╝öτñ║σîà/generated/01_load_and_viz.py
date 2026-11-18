# -*- coding: utf-8 -*-
"""
第 1 步：数据初检与初始可视化
（由起点指令 01 生成——客户分群项目第 1 件事）
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 中文显示设置（本机可用字体：Songti SC / STHeiti / PingFang SC）
matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 数据初检 ----------
df = pd.read_csv("data/mall_customers_sim.csv")
print("行列数：", df.shape)
print("\n各列缺失值：\n", df.isna().sum())
print("\n数值列基本统计：\n", df.describe().round(2))

# ---------- 初始散点图：收入-余额 ----------
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df["income"], df["balance"],
    s=(df["age"] - 18) * 8 + 20,   # 点的大小用 age 映射
    alpha=0.6, c="#4C72B0", edgecolors="white", linewidths=0.5,
)
ax.set_title("300 位零售银行客户：收入-余额分布（点越大年龄越高）", fontsize=15)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/01_raw_scatter.png", dpi=150)
print("\n已保存 outputs/01_raw_scatter.png")
