import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────────────────────────
merged = pd.read_csv('Dataset/merged_clean.csv')

merged["main_genre"] = merged["genre"].str.split(",").str[0].str.strip()
top_genres = merged["main_genre"].value_counts().head(6).index
sample = merged[merged["main_genre"].isin(top_genres)]
sample = sample[["reviewText", "scoreSentiment", "main_genre",
                  "audienceScore", "tomatoMeter"]].dropna()
sample = sample.sample(n=5000, random_state=42).reset_index(drop=True)

print(f"Sample: {len(sample):,} reviews")

colors6    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
genre_list = sorted(sample["main_genre"].unique())
genre_color = {g: colors6[i] for i, g in enumerate(genre_list)}

# ── Prepare Features ───────────────────────────────────────────────────────────
le = LabelEncoder()
sample["genre_encoded"]     = le.fit_transform(sample["main_genre"])
sample["sentiment_encoded"] = (sample["scoreSentiment"] == "POSITIVE").astype(int)

features = sample[["audienceScore", "tomatoMeter"]].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ── Genre-level profiles for dendrogram ───────────────────────────────────────
genre_profiles = sample.groupby("main_genre").agg(
    audienceScore=("audienceScore", "mean"),
    tomatoMeter  =("tomatoMeter",   "mean"),
    positive_pct =("scoreSentiment", lambda x: 100 * (x == "POSITIVE").mean())
).reset_index()

print("\nGenre profiles:")
print(genre_profiles.round(1).to_string(index=False))

X_genre = StandardScaler().fit_transform(
    genre_profiles[["audienceScore", "tomatoMeter", "positive_pct"]].values)
Z_genre = linkage(X_genre, method='ward')
order   = leaves_list(Z_genre)
sorted_genres = genre_profiles["main_genre"].values[order]


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Dendrogram
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Plot 1: Dendrogram ---")
fig, ax = plt.subplots(figsize=(12, 7))
dendrogram(
    Z_genre,
    labels=genre_profiles["main_genre"].values,
    leaf_rotation=0,
    leaf_font_size=13,
    color_threshold=1.2,
    ax=ax,
    orientation='left'
)
ax.set_title("Hierarchical Clustering Dendrogram\n"
             "How similar are genres based on audience score, tomatometer and sentiment?",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Distance (lower = more similar)", fontsize=11)
ax.set_ylabel("Genre", fontsize=11)
ax.axvline(1.2, color="red", linestyle="--", linewidth=1.5,
           label="Cut point → defines clusters")
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig("plot1_dendrogram.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Genre Profile Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Plot 2: Heatmap ---")
heatmap_data = genre_profiles.set_index("main_genre").loc[
    sorted_genres, ["audienceScore", "tomatoMeter", "positive_pct"]]
heatmap_data.columns = ["Audience Score", "Tomatometer", "% Positive Reviews"]

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto",
               vmin=0, vmax=100)

ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns, fontsize=12, fontweight="bold")
ax.set_yticks(range(len(sorted_genres)))
ax.set_yticklabels(sorted_genres, fontsize=12)

for i in range(len(sorted_genres)):
    for j in range(len(heatmap_data.columns)):
        val   = heatmap_data.values[i, j]
        color = "black" if 30 < val < 75 else "white"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

plt.colorbar(im, ax=ax, label="Score (%)", shrink=0.8)
ax.set_title("Genre Profile Heatmap\n"
             "Green = high | Red = low\n"
             "Genres sorted by dendrogram similarity",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("plot2_genre_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Hierarchical Clustering on full sample
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Running Hierarchical Clustering ---")
Z_full = linkage(X_scaled, method='ward')
n_clusters    = 3
cluster_labels = fcluster(Z_full, n_clusters, criterion='maxclust')
sample["hc_cluster"] = cluster_labels

# ── Print actual cluster profiles ─────────────────────────────────────────────
print("\nActual cluster profiles:")
cluster_means = sample.groupby("hc_cluster")[
    ["audienceScore", "tomatoMeter"]].mean()
print(cluster_means.round(1))

cluster_pos = sample.groupby("hc_cluster")["scoreSentiment"].apply(
    lambda x: 100 * (x == "POSITIVE").mean())
print("\n% Positive per cluster:")
print(cluster_pos.round(1))

# ── Name clusters based on actual tomatometer values ──────────────────────────
cluster_name_map = {}
for cid in cluster_means.index:
    tom = cluster_means.loc[cid, "tomatoMeter"]
    aud = cluster_means.loc[cid, "audienceScore"]
    pos = cluster_pos[cid]
    print(f"\nCluster {cid}: tomato={tom:.0f}%, audience={aud:.0f}%, positive={pos:.0f}%")
    if tom < 50:
        cluster_name_map[cid] = "Poorly Received"
    elif tom < 70:
        cluster_name_map[cid] = "Mixed Reception"
    else:
        cluster_name_map[cid] = "Well Received"

# Fix duplicate names by adding scores
name_counts = pd.Series(list(cluster_name_map.values())).value_counts()
if name_counts.max() > 1:
    print("\nDuplicate names detected — using score-based labels")
    for cid in cluster_means.index:
        tom = cluster_means.loc[cid, "tomatoMeter"]
        aud = cluster_means.loc[cid, "audienceScore"]
        cluster_name_map[cid] = (
            f"Cluster {cid}\n"
            f"Tomato: {tom:.0f}% | Audience: {aud:.0f}%")

print(f"\nFinal cluster names: {cluster_name_map}")
sample["cluster_name"] = sample["hc_cluster"].map(cluster_name_map)

# Order clusters from worst to best by tomatometer
cluster_order = [cluster_name_map[i]
                 for i in sorted(cluster_name_map.keys(),
                                 key=lambda x: cluster_means.loc[x, "tomatoMeter"])]
cluster_colors = ["#C44E52", "#F5C518", "#55A868"][:len(cluster_order)]

# Evaluation
sil = silhouette_score(X_scaled, sample["hc_cluster"], sample_size=1000)
db  = davies_bouldin_score(X_scaled, sample["hc_cluster"])
print(f"\nSilhouette: {sil:.3f} | Davies-Bouldin: {db:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Cluster Profiles — with n= and scores in labels
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Plot 3: Cluster Profiles ---")

aud_vals = [sample[sample["cluster_name"] == c]["audienceScore"].mean()
            for c in cluster_order]
tom_vals = [sample[sample["cluster_name"] == c]["tomatoMeter"].mean()
            for c in cluster_order]
pos_vals = [100 * (sample[sample["cluster_name"] == c]["scoreSentiment"] == "POSITIVE").mean()
            for c in cluster_order]

# Build x tick labels with n= and scores
xticklabels = []
for c in cluster_order:
    n   = (sample["cluster_name"] == c).sum()
    tom = sample[sample["cluster_name"] == c]["tomatoMeter"].mean()
    aud = sample[sample["cluster_name"] == c]["audienceScore"].mean()
    pos = 100 * (sample[sample["cluster_name"] == c]["scoreSentiment"] == "POSITIVE").mean()
    xticklabels.append(
        f"{c}\nn={n:,} | Tomato:{tom:.0f}% | Aud:{aud:.0f}% | Pos:{pos:.0f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, vals, title, ylabel in zip(
        axes,
        [aud_vals, tom_vals, pos_vals],
        ["Average Audience Score\nper Cluster",
         "Average Tomatometer\nper Cluster",
         "% Positive Sentiment\nper Cluster"],
        ["Audience Score (%)", "Tomatometer (%)", "Positive Reviews (%)"]):

    bars = ax.bar(range(len(cluster_order)), vals,
                  color=cluster_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontweight="bold", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(xticklabels, rotation=10, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    if "Sentiment" in title:
        ax.axhline(50, color="gray", linestyle="--",
                   linewidth=1.5, label="50% baseline")
        ax.legend(fontsize=9)

fig.suptitle(f"Hierarchical Clustering — Cluster Profiles (k={n_clusters})\n"
             f"Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot3_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Genre distribution per cluster — grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Plot 4: Genre per Cluster ---")

fig, ax = plt.subplots(figsize=(14, 7))

cluster_genre = pd.crosstab(sample["cluster_name"], sample["main_genre"],
                             normalize="index") * 100

x      = np.arange(len(genre_list))
width  = 0.25
offsets = np.linspace(-width, width, n_clusters)

for i, cname in enumerate(cluster_order):
    vals = [cluster_genre.loc[cname, g] if g in cluster_genre.columns else 0
            for g in genre_list]
    bars = ax.bar(x + offsets[i], vals, width,
                  label=cname, color=cluster_colors[i],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 3:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.0f}%", ha="center", fontsize=7,
                    fontweight="bold")

ax.set_title("Genre Distribution per Cluster\n"
             "→ Which genres appear most in each reception category?",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Share within Cluster (%)", fontsize=11)
ax.set_xlabel("Genre", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(genre_list, rotation=15, ha="right", fontsize=10)
ax.set_ylim(0, 55)
ax.legend(title="Cluster", fontsize=10)
ax.grid(axis="y", alpha=0.4)

for i, cname in enumerate(cluster_order):
    n   = (sample["cluster_name"] == cname).sum()
    aud = sample[sample["cluster_name"] == cname]["audienceScore"].mean()
    tom = sample[sample["cluster_name"] == cname]["tomatoMeter"].mean()
    ax.annotate(
        f"{cname}\nn={n:,} | aud={aud:.0f}% | tomato={tom:.0f}%",
        xy=(0.01 + i * 0.33, 0.97), xycoords="axes fraction",
        fontsize=8, va="top", color=cluster_colors[i], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=cluster_colors[i], alpha=0.8))

plt.tight_layout()
plt.savefig("plot4_genre_per_cluster.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Scatter — Audience Score vs Tomatometer colored by cluster
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Plot 5: Scatter Plot ---")

fig, ax = plt.subplots(figsize=(10, 7))

for i, cname in enumerate(cluster_order):
    mask = sample["cluster_name"] == cname
    n    = mask.sum()
    ax.scatter(
        sample[mask]["tomatoMeter"],
        sample[mask]["audienceScore"],
        c=cluster_colors[i], alpha=0.4, s=20,
        label=f"{cname} (n={n:,})")

ax.set_title("Audience Score vs Tomatometer\n"
             "Each dot = one review | Color = cluster",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Tomatometer — Critic Score (%)", fontsize=11)
ax.set_ylabel("Audience Score (%)", fontsize=11)
ax.plot([0, 100], [0, 100], color="gray", linestyle="--",
        linewidth=1.5, label="Perfect agreement line")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("plot5_scatter_clusters.png", dpi=150, bbox_inches="tight")
plt.show()


## ── Plot 6: Box plots with colored regions ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

data_aud = [sample[sample["cluster_name"] == c]["audienceScore"].values
            for c in cluster_order]
data_tom = [sample[sample["cluster_name"] == c]["tomatoMeter"].values
            for c in cluster_order]

for ax, data, title, ylabel in zip(
        axes,
        [data_aud, data_tom],
        ["Audience Score Distribution\nper Cluster",
         "Tomatometer Distribution\nper Cluster"],
        ["Audience Score (%)", "Tomatometer (%)"]):

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.3))

    for patch, color in zip(bp["boxes"], cluster_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set_linestyle("--")

    # Add colored background regions
    ax.axhspan(0,  40,  alpha=0.05, color="#C44E52", label="Low (0-40%)")
    ax.axhspan(40, 70,  alpha=0.05, color="#F5C518", label="Medium (40-70%)")
    ax.axhspan(70, 110, alpha=0.05, color="#55A868", label="High (70-100%)")

    # Add median + mean + n annotations BELOW each box
    for i, (d, c) in enumerate(zip(data, cluster_order)):
        median = np.median(d)
        mean   = np.mean(d)
        n      = len(d)
        pos    = 100 * (sample[sample["cluster_name"] == c]["scoreSentiment"] == "POSITIVE").mean()

        # Annotation box below x axis
        ax.text(i + 1, -15,
                f"Median: {median:.0f}%\n"
                f"Mean: {mean:.0f}%\n"
                f"Positive: {pos:.0f}%",
                ha="center", fontsize=8, fontweight="bold",
                color=cluster_colors[i],
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          edgecolor=cluster_colors[i],
                          alpha=0.9))

        # Median line label
        ax.text(i + 1, median + 1.5, f"{median:.0f}%",
                ha="center", fontsize=9, fontweight="bold",
                color="black")

    xlabels = []
    for c in cluster_order:
        n   = (sample["cluster_name"] == c).sum()
        tom = sample[sample["cluster_name"] == c]["tomatoMeter"].mean()
        aud = sample[sample["cluster_name"] == c]["audienceScore"].mean()
        xlabels.append(f"{c}\n(n={n:,} reviews)")

    ax.set_xticks(range(1, n_clusters + 1))
    ax.set_xticklabels(xlabels, rotation=0, ha="center", fontsize=10,
                       fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(-25, 110)
    ax.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Score Distributions per Cluster\n"
             "Each cluster is clearly separated by reception quality",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot6_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("HIERARCHICAL CLUSTERING SUMMARY")
print("=" * 55)
print(f"\nMethod:             Ward Linkage")
print(f"Features:           audienceScore, tomatoMeter, genre, sentiment")
print(f"Number of clusters: {n_clusters}")
print(f"Silhouette Score:   {sil:.3f}  (range -1 to 1, higher = better)")
print(f"Davies-Bouldin:     {db:.3f}  (lower = better)")
print(f"\nCluster Sizes:")
for cname in cluster_order:
    n   = (sample["cluster_name"] == cname).sum()
    tom = sample[sample["cluster_name"] == cname]["tomatoMeter"].mean()
    aud = sample[sample["cluster_name"] == cname]["audienceScore"].mean()
    pos = 100 * (sample[sample["cluster_name"] == cname]["scoreSentiment"] == "POSITIVE").mean()
    print(f"  {cname}: n={n:,} | tomato={tom:.0f}% | audience={aud:.0f}% | positive={pos:.0f}%")
print("\n✅ Done! All 6 plots saved.")