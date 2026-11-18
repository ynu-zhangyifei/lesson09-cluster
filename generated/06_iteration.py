# -*- coding: utf-8 -*-
"""
任务 06：手写 k-均值迭代过程图
不调用 KMeans.fit，自己实现「k-means++ 初始化 → 分配 → 更新质心」循环，
把第 0 轮（初始质心）、第 1 轮、第 2 轮、最后一轮画成 2×2 子图。
特征：标准化后的 income、balance；k=3；random_state=42。
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.preprocessing import StandardScaler

# ---------- 中文字体设置 ----------
候选字体 = ["Songti SC", "STHeiti", "PingFang SC",
            "Microsoft YaHei", "SimHei", "Noto Sans SC"]
已装字体 = {f.name for f in fm.fontManager.ttflist}
可用字体 = [f for f in 候选字体 if f in 已装字体]
if not 可用字体:
    raise RuntimeError("未找到可用的中文字体，请检查系统字体安装")
matplotlib.rcParams["font.sans-serif"] = 可用字体
matplotlib.rcParams["axes.unicode_minus"] = False
print("本次使用的中文字体：", 可用字体[0])

# ---------- 读入数据并标准化 ----------
df = pd.read_csv("data/mall_customers_yuan.csv", encoding="utf-8-sig")
特征 = ["income", "balance"]
X = StandardScaler().fit_transform(df[特征].to_numpy())
k = 3
rng = np.random.RandomState(42)


def kmeans_plus_plus(X, k, rng):
    """k-means++：第一个质心随机选，之后按到最近质心的距离平方概率选。"""
    已选 = [X[rng.randint(X.shape[0])]]
    for _ in range(1, k):
        最近距离平方 = np.min([((X - c) ** 2).sum(axis=1) for c in 已选], axis=0)
        概率 = 最近距离平方 / 最近距离平方.sum()
        已选.append(X[rng.choice(X.shape[0], p=概率)])
    return np.array(已选)


def 分配(X, C):
    """把每个点分给最近的质心，同时返回该分配下的 SSE（inertia）。"""
    距离平方 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    标签 = 距离平方.argmin(axis=1)
    inertia = 距离平方[np.arange(len(X)), 标签].sum()
    return 标签, inertia


def 更新质心(X, 标签, C, k):
    """每个质心移动到本簇所有点的平均位置。"""
    新C = C.copy()
    for j in range(k):
        点 = X[标签 == j]
        if len(点):
            新C[j] = 点.mean(axis=0)
    return 新C


# ---------- 迭代 ----------
C = kmeans_plus_plus(X, k, rng)
标签, inertia = 分配(X, C)
历史 = [(0, C.copy(), 标签, inertia, None)]  # (轮次, 质心, 标签, inertia, 相对上轮的移动量)

while True:
    轮次 = 历史[-1][0] + 1
    新C = 更新质心(X, 历史[-1][2], 历史[-1][1], k)
    新标签, 新inertia = 分配(X, 新C)
    移动 = float(np.linalg.norm(新C - 历史[-1][1], axis=1).sum())
    历史.append((轮次, 新C, 新标签, 新inertia, 移动))
    if 移动 < 1e-12:  # 质心不再移动 → 收敛
        break

# ---------- 终端打印每一轮 ----------
print("\n" + "=" * 62)
print("k-均值迭代过程（标准化后的 income / balance，k=3）")
print("=" * 62)
for 轮, C, 标签, inertia, 移动 in 历史:
    移动文本 = "—（初始）" if 移动 is None else f"{移动:.2f}"
    print(f"\n第 {轮} 轮：质心移动量 = {移动文本}，inertia = {inertia:.6f}")
    print(pd.DataFrame(C, columns=["income(标准化)", "balance(标准化)"]
                       ).round(4).to_string())

inertia序列 = [h[3] for h in 历史]
assert all(前 >= 后 - 1e-9 for 前, 后 in zip(inertia序列, inertia序列[1:])), \
    "inertia 出现上升，迭代实现有误"
print("\ninertia 序列：", " → ".join(f"{v:.6f}" for v in inertia序列))
print(f"共 {历史[-1][0]} 轮收敛，inertia 单调不增，最后两轮不再变化。")

# ---------- 画图 ----------
颜色表 = {0: "#DD8452", 1: "#4C72B0", 2: "#55A868"}


def 画子图(ax, 轮, C, 标签, inertia, 移动):
    for 簇 in range(k):
        mask = 标签 == 簇
        ax.scatter(X[mask, 0], X[mask, 1], s=22, alpha=0.65,
                   color=颜色表[簇], edgecolors="white", linewidths=0.35)
    ax.scatter(C[:, 0], C[:, 1], marker="*", s=520, c="red",
               edgecolors="black", linewidths=1.1, zorder=5)
    if 移动 is None:
        ax.set_title(f"第 0 轮：初始质心（k-means++）\ninertia = {inertia:.2f}",
                     fontsize=12)
    elif 移动 < 1e-12:
        ax.set_title(f"收敛：质心不再移动（第 {轮} 轮）\ninertia = {inertia:.2f}",
                     fontsize=12)
    else:
        ax.set_title(f"第 {轮} 轮：质心移动了 {移动:.2f}\ninertia = {inertia:.2f}",
                     fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlabel("年收入 income（标准化）", fontsize=10)
    ax.set_ylabel("账户余额 balance（标准化）", fontsize=10)


展示 = [历史[0], 历史[1], 历史[2]]
if 历史[-1][0] not in [h[0] for h in 展示]:
    展示.append(历史[-1])
else:  # 收敛过早（≤2 轮）时，用最后一轮替换对应展示项
    展示 = 展示[:历史[-1][0]] + [历史[-1]]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, 状态 in zip(axes.ravel(), 展示):
    画子图(ax, *状态)
for ax in axes.ravel()[len(展示):]:
    ax.axis("off")

fig.suptitle("k-均值迭代过程：分配 → 更新质心 → 再分配", fontsize=16)
plt.tight_layout()
plt.savefig("outputs/06_iteration.png", dpi=150)
print("\n已保存迭代过程图：outputs/06_iteration.png")
