# -*- coding: utf-8 -*-
"""
质心迭代过程可视化：k-均值 = Lloyd 算法（分配-更新循环）
（由指令 03 生成，手写循环，不 fit 一把梭）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/mall_customers_yuan.csv")
X = StandardScaler().fit_transform(df[["income", "balance"]])

# ---------- 手写 Lloyd 算法（k-means++ 初始化，与 KMeans 同源） ----------
def kmeanspp_init(X, K, rng):
    centers = [X[rng.integers(len(X))]]
    for _ in range(K - 1):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        prob = d2 / d2.sum()
        centers.append(X[rng.choice(len(X), p=prob)])
    return np.array(centers)

rng = np.random.default_rng(42)
K = 3
centers = kmeanspp_init(X, K, rng)

history = []          # [(centers, labels, inertia), ...]
max_iter = 20
for it in range(max_iter + 1):
    # 分配：每个点分给最近质心
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)
    inertia = ((X - centers[labels]) ** 2).sum()
    history.append((centers.copy(), labels.copy(), float(inertia)))
    print(f"第 {it} 轮：inertia={inertia:.4f}，质心=\n{np.round(centers, 3)}")
    if it == max_iter:
        break
    # 更新：质心挪到本簇平均位置
    new_centers = np.array([X[labels == k].mean(axis=0) for k in range(K)])
    shift = np.abs(new_centers - centers).max()
    if shift < 1e-8:
        break
    centers = new_centers

# ---------- 选 4 个关键时刻画 2×2 ----------
sel = [0, 1, 2, len(history) - 1]
sel = sorted(set(sel))
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
colors = ["#DD8452", "#4C72B0", "#55A868"]
for ax, hi in zip(axes.flat, sel):
    centers_h, labels_h, inertia_h = history[hi]
    for c in range(K):
        sub = X[labels_h == c]
        ax.scatter(sub[:, 0], sub[:, 1], s=25, alpha=0.6, color=colors[c])
    ax.scatter(centers_h[:, 0], centers_h[:, 1], marker="*", s=500,
               c="red", edgecolors="black", linewidths=1.2, zorder=5)
    if hi == len(history) - 1:
        title = f"第 {hi} 轮：收敛，质心不再移动（inertia={inertia_h:.1f}）"
    elif hi == 0:
        title = f"第 0 轮：k-means++ 初始质心（inertia={inertia_h:.1f}）"
    else:
        prev = history[hi - 1][0]
        shift = np.abs(centers_h - prev).max()
        title = f"第 {hi} 轮：质心移动 {shift:.2f}（inertia={inertia_h:.1f}）"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("income（标准化后）", fontsize=11)
    ax.set_ylabel("balance（标准化后）", fontsize=11)
    ax.grid(alpha=0.3)

fig.suptitle("k-均值 = 分配-更新循环：质心一步一步挪到位（inertia 单调下降）",
             fontsize=15, y=0.995)
plt.tight_layout()
plt.savefig("outputs/06_iteration.png", dpi=150)
print(f"\n共迭代 {len(history)-1} 轮收敛；已保存 outputs/06_iteration.png")
