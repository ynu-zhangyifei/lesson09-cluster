# -*- coding: utf-8 -*-
"""
选 k：肘部法 + 轮廓系数
（由修正指令 B 生成）
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

matplotlib.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "PingFang SC"]
matplotlib.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/mall_customers_yuan.csv")
X = StandardScaler().fit_transform(df[["income", "balance"]])

# ---------- 肘部法：k=1..9 的 inertia ----------
ks = range(1, 10)
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
print("肘部法 inertia 表：")
for k, v in zip(ks, inertias):
    print(f"  k={k}: inertia={v:.2f}")

# ---------- 轮廓系数：k=2..9 ----------
ks2 = range(2, 10)
sils = []
for k in ks2:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sils.append(silhouette_score(X, labels))
print("\n轮廓系数表：")
for k, v in zip(ks2, sils):
    print(f"  k={k}: silhouette={v:.4f}")

# ---------- 并排双折线 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
ax.plot(list(ks), inertias, "o-", color="#4C72B0", linewidth=2)
ax.axvline(x=3, color="red", linestyle="--", alpha=0.6, label="拐点 k=3")
for k, v in zip(ks, inertias):
    ax.annotate(f"{v:.0f}", (k, v), textcoords="offset points", xytext=(0, 8),
                fontsize=9, ha="center")
ax.set_title("肘部法：簇内平方和（inertia）随 k 的变化", fontsize=13)
ax.set_xlabel("簇数 k", fontsize=11)
ax.set_ylabel("簇内平方和 inertia", fontsize=11)
ax.set_xticks(list(ks))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(list(ks2), sils, "s-", color="#55A868", linewidth=2)
best_k = list(ks2)[sils.index(max(sils))]
ax.axvline(x=best_k, color="red", linestyle="--", alpha=0.6, label=f"最高点 k={best_k}")
for k, v in zip(ks2, sils):
    ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 8),
                fontsize=9, ha="center")
ax.set_title("轮廓系数：分得好不好（越大越好，最大 1）", fontsize=13)
ax.set_xlabel("簇数 k", fontsize=11)
ax.set_ylabel("轮廓系数 silhouette", fontsize=11)
ax.set_xticks(list(ks2))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/05_choose_k.png", dpi=150)
print("\n已保存 outputs/05_choose_k.png")

print(f"\n结论：肘部法拐点 k=3；轮廓系数最高 k={best_k}（{max(sils):.4f}），"
      f"k=3 时轮廓系数为 {sils[1]:.4f}——两个证据不完全一致：k=2 分得最开"
      f"（年轻客群 vs 其余），k=3 才把中产与富裕拆开。选 k 最终是业务决策："
      f"零售部需要三类客群三种打法，故取 k=3。")
