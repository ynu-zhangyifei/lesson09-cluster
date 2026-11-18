# -*- coding: utf-8 -*-
"""
修正 A + 修正 B 合并脚本
数据源：data/mall_customers_yuan.csv（balance 以元计，演示标准化前后对照）
修正 A：标准化 + KMeans(k=3, rs=42, n_init=10) → 左右对照散点图 outputs/04_compare_scale.png
修正 B：k=1..9 扫肘部法 + k=2..9 扫轮廓系数 → 并排折线图 outputs/05_choose_k.png
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------- 中文字体 ----------
候选字体 = ["Songti SC", "STHeiti", "PingFang SC",
            "Microsoft YaHei", "SimHei", "Noto Sans SC"]
已装字体 = {f.name for f in fm.fontManager.ttflist}
可用字体 = [f for f in 候选字体 if f in 已装字体]
matplotlib.rcParams["font.sans-serif"] = 可用字体
matplotlib.rcParams["axes.unicode_minus"] = False
print("中文字体：", 可用字体[0])

df = pd.read_csv("data/mall_customers_yuan.csv", encoding="utf-8-sig")
特征 = ["income", "balance"]
X_raw = df[特征].to_numpy()
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)
k = 3
颜色表 = {0: "#DD8452", 1: "#4C72B0", 2: "#55A868"}

# ==========================================================
# 修正 A：标准化后聚类 + 原始 vs 标准化坐标并排对照
# ==========================================================
print("\n" + "=" * 60)
print("修正 A：标准化后 KMeans(k=3, random_state=42, n_init=10)")
print("=" * 60)

# 标准化空间聚类（修正后）
km_std = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_std)
标签_std = km_std.labels_
质心_raw_std = scaler.inverse_transform(km_std.cluster_centers_)

# 未标准化聚类（翻车版，k=3，rs=42，n_init=10）
km_raw = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_raw)
标签_raw = km_raw.labels_

修正前人数 = pd.Series(标签_raw).value_counts().sort_index().to_dict()
修正后人数 = pd.Series(标签_std).value_counts().sort_index().to_dict()
print(f"修正前（未标准化）各簇人数：{修正前人数}")
print(f"修正后（标准化）各簇人数：{修正后人数}")

# 画像对比
画像_std = pd.DataFrame(
    [df.loc[标签_std == j, 特征].mean().round(2).to_dict() for j in range(k)],
    index=[f"簇 {j}" for j in range(k)]
)
画像_std["人数"] = [int((标签_std == j).sum()) for j in range(k)]
画像_std.index.name = "修正后画像"
print("\n修正后各簇画像（均值）：")
print(画像_std.round(2).to_string())

# 画图：左右两个子图，原始+标准化着色
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for j in range(k):
    mask = 标签_std == j
    # 左图：原始坐标，着色 = 标准化空间分配结果
    axes[0].scatter(X_raw[mask, 0], X_raw[mask, 1],
                    s=22, alpha=0.65, color=颜色表[j],
                    edgecolors="white", linewidths=0.35,
                    label=f"簇 {j}（{mask.sum()} 人）")
    # 质心画回原始坐标（逆变换）
    axes[0].scatter(质心_raw_std[j, 0], 质心_raw_std[j, 1],
                    marker="*", s=520, c="red", edgecolors="black",
                    linewidths=1.1, zorder=5)
    # 右图：标准化坐标
    axes[1].scatter(X_std[mask, 0], X_std[mask, 1],
                    s=22, alpha=0.65, color=颜色表[j],
                    edgecolors="white", linewidths=0.35)
    axes[1].scatter(km_std.cluster_centers_[j, 0], km_std.cluster_centers_[j, 1],
                    marker="*", s=520, c="red", edgecolors="black",
                    linewidths=1.1, zorder=5)

axes[0].set_title("左：原始坐标（balance 单位元），着色来自标准化空间的分配", fontsize=11)
axes[0].set_xlabel("年收入 income（万元）")
axes[0].set_ylabel("账户平均余额 balance（元）")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].set_title("右：标准化坐标（两特征等权参与距离计算）", fontsize=11)
axes[1].set_xlabel("income（标准化）")
axes[1].set_ylabel("balance（标准化）")
axes[1].grid(alpha=0.3)

fig.suptitle("量纲陷阱修正：标准化不是改变客户，是改变度量衡", fontsize=15)
plt.tight_layout()
plt.savefig("outputs/04_compare_scale.png", dpi=150)
print("\n已保存：outputs/04_compare_scale.png")

# ==========================================================
# 修正 B：选 k —— 肘部法 + 轮廓系数
# ==========================================================
print("\n" + "=" * 60)
print("修正 B：选 k 双证据（肘部法 + 轮廓系数）")
print("=" * 60)

ks = list(range(1, 10))
inertias = []
silhouettes = []

print("\n肘部法（k=1..9）数值表：")
for kk in ks:
    km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X_std)
    inertias.append(km.inertia_)
    print(f"  k={kk:2d}  inertia = {km.inertia_:10.2f}")

print("\n轮廓系数（k=2..9）数值表：")
for kk in ks[1:]:
    km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X_std)
    sil = silhouette_score(X_std, km.labels_)
    silhouettes.append(sil)
    print(f"  k={kk:2d}  silhouette = {sil:.4f}")

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(ks, inertias, marker="o", linewidth=2, color="#4C72B0")
# 标注拐点：观察数值 k=3 处下降变缓
axes[0].scatter([3], [inertias[2]], s=200, c="red", zorder=5)
axes[0].annotate(f"肘部拐点 k=3\ninertia={inertias[2]:.1f}",
                 xy=(3, inertias[2]), xytext=(3.5, inertias[2] + 100),
                 arrowprops=dict(arrowstyle="->"), fontsize=10, color="red")
axes[0].set_title("肘部法：inertia 随 k 变化", fontsize=12)
axes[0].set_xlabel("簇数 k")
axes[0].set_ylabel("簇内平方和 inertia")
axes[0].grid(alpha=0.3)
axes[0].set_xticks(ks)

axes[1].plot(ks[1:], silhouettes, marker="o", linewidth=2, color="#DD8452")
# 找最高轮廓系数
最佳k = ks[1:][np.argmax(silhouettes)]
axes[1].scatter([最佳k], [max(silhouettes)], s=200, c="red", zorder=5)
axes[1].annotate(f"最高 k={最佳k}\n轮廓系数={max(silhouettes):.4f}",
                 xy=(最佳k, max(silhouettes)), xytext=(最佳k + 0.6, max(silhouettes) - 0.03),
                 arrowprops=dict(arrowstyle="->"), fontsize=10, color="red")
axes[1].set_title("轮廓系数：群间距离 vs 群内紧密程度", fontsize=12)
axes[1].set_xlabel("簇数 k")
axes[1].set_ylabel("silhouette_score")
axes[1].grid(alpha=0.3)
axes[1].set_xticks(ks[1:])

fig.suptitle("选 k 双证据：数学给证据，业务来拍板", fontsize=15)
plt.tight_layout()
plt.savefig("outputs/05_choose_k.png", dpi=150)
print("\n已保存：outputs/05_choose_k.png")
print(f"\n选 k 结论：肘部法拐点 k=3；轮廓系数最高点 k={最佳k}。")
print("若两证据不一致，业务决策优先——银行要三套打法（年轻人/中产/高净值），取 k=3。")
