import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/Users/nourelrahman/PycharmProjects/SEP-Project-')


print("Loading full dataset...")
merged = pd.read_csv('Dataset/merged_clean.csv')
merged["main_genre"] = merged["genre"].str.split(",").str[0].str.strip()
top_genres = merged["main_genre"].value_counts().head(6).index
df = merged[merged["main_genre"].isin(top_genres)]
df = df[["scoreSentiment", "main_genre",
         "audienceScore", "tomatoMeter"]].dropna().reset_index(drop=True)

print(f"Full dataset: {len(df):,} reviews")
print(f"Genres: {list(df['main_genre'].unique())}")

genre_list   = sorted(df["main_genre"].unique())
genre_colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
genre_color  = {g: genre_colors[i] for i, g in enumerate(genre_list)}



# GENRE-BASED ANALYSIS

print(" " + "="*55)
print("PART 1: Genre-based Analysis")
print("="*55)

genre_profiles = df.groupby("main_genre").agg(
    audienceScore      =("audienceScore", "mean"),
    tomatoMeter        =("tomatoMeter",   "mean"),
    positive_pct       =("scoreSentiment", lambda x: 100*(x=="POSITIVE").mean()),
    critic_audience_gap=("audienceScore",  lambda x:
                         x.mean() - df.loc[x.index, "tomatoMeter"].mean()),
    n=("audienceScore", "count")
).reset_index().sort_values("tomatoMeter", ascending=False)

genre_profiles["negative_pct"] = 100 - genre_profiles["positive_pct"]
sorted_genres = genre_profiles["main_genre"].values

print("Genre profiles:")
print(genre_profiles.round(1).to_string(index=False))


# plot 1: Critic vs Audience Score per Genre
print(" Plot 1: Scores per Genre ")

fig, ax = plt.subplots(figsize=(13, 7))

aud_vals = [genre_profiles[genre_profiles["main_genre"]==g]["audienceScore"].values[0]
            for g in sorted_genres]
tom_vals = [genre_profiles[genre_profiles["main_genre"]==g]["tomatoMeter"].values[0]
            for g in sorted_genres]
n_vals   = [genre_profiles[genre_profiles["main_genre"]==g]["n"].values[0]
            for g in sorted_genres]

x     = np.arange(len(sorted_genres))
width = 0.35

bars1 = ax.bar(x - width/2, tom_vals, width,
               label="Tomatometer (Critic Score)",
               color="#DD8452", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, aud_vals, width,
               label="Audience Score",
               color="#4C72B0", edgecolor="black", linewidth=0.5)

for bar, val in zip(bars1, tom_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")
for bar, val in zip(bars2, aud_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")


for i, (tom, aud) in enumerate(zip(tom_vals, aud_vals)):
    diff  = aud - tom
    color = "#2d6a4f" if diff > 0 else "#7d1a1a"
    ax.annotate(f"Δ{diff:+.1f}%",
                xy=(i, max(tom, aud)+3),
                ha="center", fontsize=9,
                color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{g}\n(n={n:,})" for g, n in zip(sorted_genres, n_vals)],
    fontsize=10, fontweight="bold")
ax.set_title("Critic Score vs Audience Score per Genre\n"
             "Orange = Tomatometer (Critics) | Blue = Audience Score\n"
             "Δ = difference (green = audiences rate higher | red = critics rate higher)",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Average Score (%)", fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("plot1_scores_per_genre.png", dpi=150, bbox_inches="tight")
plt.show()


#plot 2: Positive vs Negative Reviews per Genre
print(" Plot 2: Positive vs Negative ")

fig, ax = plt.subplots(figsize=(13, 6))

pos_vals = [genre_profiles[genre_profiles["main_genre"]==g]
            ["positive_pct"].values[0] for g in sorted_genres]
neg_vals = [100 - p for p in pos_vals]

x     = np.arange(len(sorted_genres))
width = 0.35

bars1 = ax.bar(x - width/2, pos_vals, width,
               label="Positive Reviews (%)",
               color="#55A868", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, neg_vals, width,
               label="Negative Reviews (%)",
               color="#C44E52", edgecolor="black", linewidth=0.5)

for bar, val in zip(bars1, pos_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.0f}%", ha="center", fontsize=11,
            fontweight="bold", color="#2d6a4f")
for bar, val in zip(bars2, neg_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.0f}%", ha="center", fontsize=11,
            fontweight="bold", color="#7d1a1a")

ax.axhline(50, color="gray", linestyle="--",
           linewidth=1.5, label="50% baseline")
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{g}\n(n={genre_profiles[genre_profiles['main_genre']==g]['n'].values[0]:,})"
     for g in sorted_genres],
    fontsize=10, fontweight="bold")
ax.set_title("Positive vs Negative Reviews per Genre\n"
             "→ Which genres are most liked and most disliked?",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Percentage of Reviews (%)", fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("plot2_positive_negative.png", dpi=150, bbox_inches="tight")
plt.show()

#plot 3: Genre Heatmap
print(" Plot 3: Genre Heatmap ")

heatmap_cols   = ["tomatoMeter", "audienceScore",
                  "positive_pct", "negative_pct"]
heatmap_labels = ["Tomatometer", "Audience Score",
                  "% Positive Reviews", "% Negative Reviews"]

heatmap_data = genre_profiles.set_index("main_genre").loc[
    sorted_genres, heatmap_cols].values


display_data = heatmap_data.copy()
display_data[:, 3] = 100 - display_data[:, 3]

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(display_data, cmap="RdYlGn",
               aspect="auto", vmin=0, vmax=100)

ax.set_xticks(range(len(heatmap_labels)))
ax.set_xticklabels(heatmap_labels, fontsize=12, fontweight="bold")
ax.tick_params(axis='x', pad=10)

ax.set_yticks(range(len(sorted_genres)))
ax.set_yticklabels(sorted_genres, fontsize=12)


for i in range(len(sorted_genres)):
    for j in range(len(heatmap_labels)):
        val = heatmap_data[i, j]
        # Text color based on display value
        disp_val = display_data[i, j]
        color = "black" if 25 < disp_val < 75 else "white"
        ax.text(j, i, f"{val:.0f}%",
                ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)


for x_pos in [0.5, 1.5, 2.5]:
    ax.axvline(x_pos, color="white", linewidth=2)


for y_pos in np.arange(0.5, len(sorted_genres)-0.5, 1):
    ax.axhline(y_pos, color="white", linewidth=1)


ax.annotate("← Higher is better",
            xy=(1.0, len(sorted_genres)-0.3),
            xycoords=("data", "data"),
            ha="center", fontsize=9,
            color="gray", style="italic")
ax.annotate("Higher = more negative →",
            xy=(3.0, len(sorted_genres)-0.3),
            xycoords=("data", "data"),
            ha="center", fontsize=9,
            color="gray", style="italic")

plt.colorbar(im, ax=ax, label="Score (%)", shrink=0.8)
ax.set_title("Genre Profile Overview\n"
             "Green = good | Red = bad | "
             "% Negative column: red = many negative reviews",
             fontweight="bold", fontsize=13)


plt.subplots_adjust(left=0.18)
plt.tight_layout()
plt.savefig("plot3_genre_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# ML CLUSTERING

print(" " + "="*55)
print("PART 2: ML Clustering")
print("="*55)

features = df[["audienceScore", "tomatoMeter"]].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)

print("Evaluating k values...")
sil_scores = []
db_scores  = []
k_range    = range(2, 8)

for k in k_range:
    mbk    = MiniBatchKMeans(n_clusters=k, random_state=42,
                              batch_size=10000, n_init=10)
    labels = mbk.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=10000))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    print(f"  k={k}: Silhouette={sil_scores[-1]:.3f}, DB={db_scores[-1]:.3f}")

best_k = 4
print(f"Using k={best_k}")

mbk_final = MiniBatchKMeans(n_clusters=best_k, random_state=42,
                             batch_size=10000, n_init=10)
df["cluster"] = mbk_final.fit_predict(X_scaled) + 1

cluster_means = df.groupby("cluster")[["audienceScore","tomatoMeter"]].mean()
cluster_pos   = df.groupby("cluster")["scoreSentiment"].apply(
    lambda x: 100*(x=="POSITIVE").mean())


cluster_name_map = {}
for cid in cluster_means.index:
    tom = cluster_means.loc[cid, "tomatoMeter"]
    if tom < 35:
        cluster_name_map[cid] = "Poorly Received"
    elif tom < 55:
        cluster_name_map[cid] = "Below Average"
    elif tom < 75:
        cluster_name_map[cid] = "Above Average"
    else:
        cluster_name_map[cid] = "Well Received"


name_counts = pd.Series(list(cluster_name_map.values())).value_counts()
if name_counts.max() > 1:
    for cid in cluster_means.index:
        tom = cluster_means.loc[cid, "tomatoMeter"]
        aud = cluster_means.loc[cid, "audienceScore"]
        cluster_name_map[cid] = (
            f"Cluster {cid}\n"
            f"Tomato:{tom:.0f}% | Audience:{aud:.0f}%")

df["cluster_name"] = df["cluster"].map(cluster_name_map)
cluster_order  = [cluster_name_map[i]
                  for i in sorted(cluster_name_map.keys(),
                                  key=lambda x: cluster_means.loc[x,"tomatoMeter"])]
cluster_colors = ["#C44E52","#DD8452","#F5C518","#55A868"]

sil = silhouette_score(X_scaled, df["cluster"].values, sample_size=10000)
db  = davies_bouldin_score(X_scaled, df["cluster"].values)
print(f"Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f}")

print("Cluster profiles:")
for cname in cluster_order:
    n   = (df["cluster_name"]==cname).sum()
    tom = df[df["cluster_name"]==cname]["tomatoMeter"].mean()
    aud = df[df["cluster_name"]==cname]["audienceScore"].mean()
    pos = 100*(df[df["cluster_name"]==cname]["scoreSentiment"]=="POSITIVE").mean()
    print(f"  {cname.split(chr(10))[0]}: "
          f"n={n:,} | tomato={tom:.0f}% | audience={aud:.0f}% | positive={pos:.0f}%")


#plot 4: Evaluation Metrics
print("Plot 4: Evaluation Metrics ")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(k_range, sil_scores, marker='o', color="#55A868",
         linewidth=2.5, markersize=8)
ax1.fill_between(k_range, sil_scores, alpha=0.1, color="#55A868")
ax1.axvline(best_k, color="red", linestyle="--", linewidth=2,
            label=f"Chosen k={best_k}")
for k, s in zip(k_range, sil_scores):
    ax1.text(k, s+0.005, f"{s:.3f}", ha="center", fontsize=8)
ax1.set_title("Silhouette Score per k\n(higher = better separated clusters)",
              fontweight="bold", fontsize=12)
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Silhouette Score")
ax1.legend()
ax1.grid(alpha=0.4)

ax2.plot(k_range, db_scores, marker='o', color="#C44E52",
         linewidth=2.5, markersize=8)
ax2.fill_between(k_range, db_scores, alpha=0.1, color="#C44E52")
ax2.axvline(best_k, color="red", linestyle="--", linewidth=2,
            label=f"Chosen k={best_k}")
for k, d in zip(k_range, db_scores):
    ax2.text(k, d+0.01, f"{d:.3f}", ha="center", fontsize=8)
ax2.set_title("Davies-Bouldin Score per k\n(lower = more compact clusters)",
              fontweight="bold", fontsize=12)
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Davies-Bouldin Score")
ax2.legend()
ax2.grid(alpha=0.4)

fig.suptitle(f"ML Clustering — How Many Clusters Are Optimal?\n"
             f"→ k={best_k} chosen: Silhouette={sil_scores[best_k-2]:.3f} | "
             f"Davies-Bouldin={db_scores[best_k-2]:.3f}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot4_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()


#plot 5: Cluster Profiles
print("Plot 5: Cluster Profiles")

aud_vals = [df[df["cluster_name"]==c]["audienceScore"].mean()
            for c in cluster_order]
tom_vals = [df[df["cluster_name"]==c]["tomatoMeter"].mean()
            for c in cluster_order]
pos_vals = [100*(df[df["cluster_name"]==c]["scoreSentiment"]=="POSITIVE").mean()
            for c in cluster_order]
neg_vals = [100 - p for p in pos_vals]

xticklabels = []
for c in cluster_order:
    n   = (df["cluster_name"]==c).sum()
    tom = df[df["cluster_name"]==c]["tomatoMeter"].mean()
    aud = df[df["cluster_name"]==c]["audienceScore"].mean()
    pos = 100*(df[df["cluster_name"]==c]["scoreSentiment"]=="POSITIVE").mean()
    xticklabels.append(
        f"{c.split(chr(10))[0]}\nn={n:,}\nTomato:{tom:.0f}% | Aud:{aud:.0f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, vals, title, ylabel in zip(
        axes,
        [aud_vals, tom_vals, pos_vals],
        ["Average Audience Score\nper Cluster",
         "Average Tomatometer\nper Cluster",
         "% Positive Sentiment\nper Cluster"],
        ["Audience Score (%)", "Tomatometer (%)", "Positive Reviews (%)"]):

    bars = ax.bar(range(len(cluster_order)), vals,
                  color=cluster_colors, edgecolor="black",
                  linewidth=0.5, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.0f}%", ha="center", fontweight="bold", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(xticklabels, rotation=5, ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    if "Sentiment" in title:
        ax.axhline(50, color="gray", linestyle="--",
                   linewidth=1.5, label="50% baseline")
        ax.legend(fontsize=9)

fig.suptitle(f"ML Cluster Profiles — Reception Quality\n"
             f"Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f} | "
             f"Total: {len(df):,} reviews",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot5_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.show()


#plot 6: clusters vs Genre
print("Plot 6: Scatter Plot ")

scatter_sample = df.sample(n=15000, random_state=42)
distinct_genre_colors = ["#e41a1c","#377eb8","#4daf4a",
                         "#984ea3","#ff7f00","#a65628"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))


for i, cname in enumerate(cluster_order):
    mask = scatter_sample["cluster_name"] == cname
    n    = (df["cluster_name"]==cname).sum()
    ax1.scatter(
        scatter_sample[mask]["tomatoMeter"],
        scatter_sample[mask]["audienceScore"],
        c=cluster_colors[i], alpha=0.5, s=15,
        label=f"{cname.split(chr(10))[0]} (n={n:,})")


for i, cname in enumerate(cluster_order):
    cid = [k for k,v in cluster_name_map.items() if v==cname][0]
    tom = cluster_means.loc[cid,"tomatoMeter"]
    aud = cluster_means.loc[cid,"audienceScore"]
    ax1.scatter(tom, aud, c=cluster_colors[i], s=300,
                marker="*", edgecolor="black", linewidth=1.5, zorder=5)
    ax1.annotate(f" {cname.split(chr(10))[0]}",
                 (tom, aud), fontsize=8, fontweight="bold",
                 color=cluster_colors[i])

ax1.plot([0,100],[0,100], color="black", linestyle="--",
         linewidth=1.5, label="Perfect agreement")
ax1.set_title("Colored by ML Cluster\n★ = cluster centroid",
              fontweight="bold", fontsize=12)
ax1.set_xlabel("Tomatometer — Critic Score (%)", fontsize=11)
ax1.set_ylabel("Audience Score (%)", fontsize=11)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(alpha=0.3)


for i, genre in enumerate(genre_list):
    mask = scatter_sample["main_genre"] == genre
    n    = (df["main_genre"]==genre).sum()
    ax2.scatter(
        scatter_sample[mask]["tomatoMeter"],
        scatter_sample[mask]["audienceScore"],
        c=distinct_genre_colors[i], alpha=0.5, s=15,
        label=f"{genre} (n={n:,})")

ax2.plot([0,100],[0,100], color="black", linestyle="--",
         linewidth=1.5, label="Perfect agreement")
ax2.set_title("Colored by Genre",
              fontweight="bold", fontsize=12)
ax2.set_xlabel("Tomatometer — Critic Score (%)", fontsize=11)
ax2.set_ylabel("Audience Score (%)", fontsize=11)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(alpha=0.3)

fig.suptitle("Scatter: Critic vs Audience Score\n"
             "Left = ML Clusters | Right = Genres\n"
             "→ ML clusters separate clearly | Genres spread across all score ranges",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot6_scatter_comparison.png", dpi=150, bbox_inches="tight")
plt.show()



# SUMMARY

print(" " + "="*55)
print("FULL DATASET CLUSTERING SUMMARY")
print("="*55)

print(f"PART 1: Genre Analysis ")
print(f"Total reviews: {len(df):,}")
for _, row in genre_profiles.iterrows():
    gap       = row["audienceScore"] - row["tomatoMeter"]
    direction = "audiences rate higher" if gap > 0 else "critics rate higher"
    print(f"  {row['main_genre']}: tomato={row['tomatoMeter']:.0f}%, "
          f"audience={row['audienceScore']:.0f}%, "
          f"positive={row['positive_pct']:.0f}%, "
          f"negative={row['negative_pct']:.0f}%, "
          f"gap={gap:+.1f}% ({direction})")

print(f" PART 2: ML Clustering ")
print(f"Algorithm:      MiniBatchKMeans")
print(f"Features:       audienceScore, tomatoMeter")
print(f"Total reviews:  {len(df):,}")
print(f"Clusters:       {best_k}")
print(f"Silhouette:     {sil:.3f}")
print(f"Davies-Bouldin: {db:.3f}")
for cname in cluster_order:
    n   = (df["cluster_name"]==cname).sum()
    tom = df[df["cluster_name"]==cname]["tomatoMeter"].mean()
    aud = df[df["cluster_name"]==cname]["audienceScore"].mean()
    pos = 100*(df[df["cluster_name"]==cname]["scoreSentiment"]=="POSITIVE").mean()
    neg = 100 - pos
    print(f"\n  {cname.split(chr(10))[0]}:")
    print(f"    n              = {n:,}")
    print(f"    Tomatometer    = {tom:.0f}%")
    print(f"    Audience Score = {aud:.0f}%")
    print(f"    Positive       = {pos:.0f}%")
    print(f"    Negative       = {neg:.0f}%")

print("Done! All 6 plots saved.")