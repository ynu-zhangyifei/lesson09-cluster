# -*- coding: utf-8 -*-
"""
任务 1：数据初检与初始可视化
读入客户数据，检查行列数 / 缺失值 / 数值列统计，
并画出「收入-余额」散点图（点的大小映射年龄）。
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---------- 中文字体设置 ----------
# 优先使用本机可用的中文字体，逐个检测，避免出现方框乱码
候选字体 = ["Songti SC", "STHeiti", "PingFang SC",
            "Microsoft YaHei", "SimHei", "Noto Sans SC"]
已装字体 = {f.name for f in fm.fontManager.ttflist}
可用字体 = [f for f in 候选字体 if f in 已装字体]
if not 可用字体:
    raise RuntimeError("未找到可用的中文字体，请检查系统字体安装")
matplotlib.rcParams["font.sans-serif"] = 可用字体
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
print("本次使用的中文字体：", 可用字体[0])

# ---------- 1. 读入数据 ----------
df = pd.read_csv("data/mall_customers_sim.csv", encoding="utf-8-sig")

print("\n" + "=" * 55)
print("一、数据初检")
print("=" * 55)
print(f"行列数：{df.shape[0]} 行 × {df.shape[1]} 列")
print("\n各列缺失值：")
print(df.isna().sum().to_string())

数值列 = ["age", "income", "balance", "product_count"]
print("\n数值列基本统计：")
print(df[数值列].describe().round(2).to_string())

# ---------- 2. 收入-余额散点图 ----------
# 把年龄线性映射到点面积，保证最小 25、最大 200，肉眼可分辨
点大小 = 25 + (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min()) * 175

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df["income"], df["balance"],
           s=点大小, c="#4C72B0", alpha=0.6,
           edgecolors="white", linewidths=0.5)

ax.set_title("零售银行客户收入-余额分布（点越大表示年龄越高）", fontsize=16)
ax.set_xlabel("年收入 income（万元）", fontsize=12)
ax.set_ylabel("账户平均余额 balance（万元）", fontsize=12)
ax.grid(alpha=0.3)

# 用两个空点做「点大小=年龄」的图例说明
图例 = [ax.scatter([], [], s=25 + (18 - df["age"].min()) / (df["age"].max() - df["age"].min()) * 175,
                   c="#4C72B0", alpha=0.6, edgecolors="white", label="18 岁左右"),
        ax.scatter([], [], s=25 + (63 - df["age"].min()) / (df["age"].max() - df["age"].min()) * 175,
                   c="#4C72B0", alpha=0.6, edgecolors="white", label="63 岁左右")]
ax.legend(handles=图例, title="客户年龄", loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/01_raw_scatter.png", dpi=150)
print("\n已保存初始散点图：outputs/01_raw_scatter.png")
