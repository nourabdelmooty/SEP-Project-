import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

merged = pd.read_csv('Dataset/merged_clean.csv')
emotions = pd.read_csv('Dataset/classification_sample_100k_with_emotions.csv')


merged["main_genre"] = merged["genre"].str.split(",").str[0].str.strip()
top_genres = merged["main_genre"].value_counts().head(6).index
merged = merged[merged["main_genre"].isin(top_genres)]

df = emotions.merge(
    merged[["reviewId", "main_genre", "audienceScore", "tomatoMeter"]],
    on="reviewId", how="inner")
df = df.dropna(subset=["joy", "anger", "sadness", "fear",
                        "disgust", "surprise", "neutral",
                        "emotional_intensity", "audienceScore",
                        "tomatoMeter", "main_genre"]).reset_index(drop=True)

print(f"Dataset: {len(df):,} reviews")
print(f"Genres: {list(df['main_genre'].unique())}")

genre_list    = sorted(df["main_genre"].unique())
emotions_plot = ["joy", "anger", "sadness", "fear", "disgust", "surprise"]
em_colors     = {
    "joy":     "#F5C518",
    "anger":   "#C44E52",
    "sadness": "#4C72B0",
    "fear":    "#8172B2",
    "disgust": "#937860",
    "surprise":"#DD8452"
}
genre_colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
genre_color  = {g: genre_colors[i] for i, g in enumerate(genre_list)}



print(" " + "="*60)
print("PART 1: Genre-based Emotion Analysis")
print("="*60)

genre_emotion = df.groupby("main_genre")[emotions_plot].mean()
genre_scores  = df.groupby("main_genre").agg(
    audienceScore=("audienceScore", "mean"),
    tomatoMeter  =("tomatoMeter",   "mean"),
    positive_pct =("scoreSentiment", lambda x: 100*(x=="POSITIVE").mean()),
    emotional_intensity=("emotional_intensity", "mean"),
    n            =("reviewText", "count")
).reset_index()

print("Genre emotion profiles:")
print(pd.concat([genre_emotion.round(3),
                 genre_scores.set_index("main_genre")[
                     ["tomatoMeter","positive_pct"]].round(1)], axis=1))


#plot 1: Emotion scores per genre
print(" Plot 1: Emotions per Genre ")

fig, ax = plt.subplots(figsize=(14, 7))
x       = np.arange(len(genre_list))
width   = 0.8 / len(emotions_plot)
offsets = np.linspace(-width*(len(emotions_plot)-1)/2,
                       width*(len(emotions_plot)-1)/2,
                       len(emotions_plot))

for i, emotion in enumerate(emotions_plot):
    vals = [genre_emotion.loc[g, emotion] for g in genre_list]
    bars = ax.bar(x + offsets[i], vals, width,
                  label=emotion.capitalize(),
                  color=em_colors[emotion],
                  edgecolor="black", linewidth=0.3)
    for bar, val in zip(bars, vals):
        if val > 0.07:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.003,
                    f"{val:.2f}", ha="center",
                    fontsize=6.5, fontweight="bold")

ax.set_title("Average Emotion Score per Genre\n"
             "→ Do different genres trigger different emotions in reviewers?\n"
             "(0 = emotion absent | 1 = emotion dominant)",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Emotion Score (0–1)", fontsize=11)
ax.set_xlabel("Genre", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{g}\n(n={genre_scores[genre_scores['main_genre']==g]['n'].values[0]:,})"
     for g in genre_list],
    fontsize=10, fontweight="bold")
ax.set_ylim(0, 0.5)
ax.legend(title="Emotion", fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("plot1_emotions_per_genre.png", dpi=150, bbox_inches="tight")
plt.show()


# plot 2: Which genre has the highest score per emotion
print(" Plot 2: Dominant Genre per Emotion ")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, emotion in enumerate(emotions_plot):
    vals   = [genre_emotion.loc[g, emotion] for g in genre_list]
    colors = [genre_color[g] for g in genre_list]
    sorted_idx = np.argsort(vals)
    sorted_genres = [genre_list[j] for j in sorted_idx]
    sorted_vals   = [vals[j] for j in sorted_idx]
    sorted_colors = [colors[j] for j in sorted_idx]

    bars = axes[i].barh(sorted_genres, sorted_vals,
                        color=sorted_colors, edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, sorted_vals):
        axes[i].text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    axes[i].set_title(f"{emotion.capitalize()}\n"
                      f"→ Which genre scores highest?",
                      fontweight="bold", fontsize=11,
                      color=em_colors[emotion])
    axes[i].set_xlabel("Emotion Score (0–1)", fontsize=9)
    axes[i].set_xlim(0, max(sorted_vals)*1.3)
    axes[i].grid(axis="x", alpha=0.4)
    axes[i].axvline(np.mean(sorted_vals), color="gray",
                    linestyle="--", linewidth=1,
                    label=f"Mean: {np.mean(sorted_vals):.3f}")
    axes[i].legend(fontsize=8)

fig.suptitle("Which Genre Triggers Each Emotion the Most?\n"
             "→ Each subplot shows one emotion ranked by genre",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot2_dominant_genre_per_emotion.png", dpi=150, bbox_inches="tight")
plt.show()


#plot 3: genres vs emotions + scores
print(" Plot 3: Genre Heatmap ")

heatmap_cols   = ["tomatoMeter", "audienceScore", "positive_pct",
                  "emotional_intensity"]
heatmap_labels = ["Tomatometer", "Audience Score", "% Positive",
                  "Emotional\nIntensity"]

emotion_scaled = genre_emotion * 100
score_data     = genre_scores.set_index("main_genre")[heatmap_cols]
score_data["emotional_intensity"] = score_data["emotional_intensity"] * 100

all_data = pd.concat([score_data, emotion_scaled], axis=1)
all_data_sorted = all_data.sort_values("tomatoMeter", ascending=True)

all_labels = heatmap_labels + [e.capitalize() for e in emotions_plot]

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(all_data_sorted.values, cmap="RdYlGn",
               aspect="auto", vmin=0, vmax=100)

ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels(all_labels, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(all_data_sorted)))
ax.set_yticklabels(all_data_sorted.index, fontsize=12)

for i in range(len(all_data_sorted)):
    for j in range(len(all_labels)):
        val   = all_data_sorted.values[i, j]
        color = "black" if 25 < val < 70 else "white"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)


ax.axvline(3.5, color="white", linewidth=3)
ax.text(1.5, -0.7, "← Scores →", ha="center", fontsize=10,
        color="gray", style="italic")
ax.text(7.5, -0.7, "← Emotion Scores →", ha="center", fontsize=10,
        color="gray", style="italic")

plt.colorbar(im, ax=ax, label="Score (%)", shrink=0.8)
ax.set_title("Genre Profile: Scores & Emotions\n"
             "Green = high | Red = low | Sorted by Tomatometer\n"
             "Left = review scores | Right = emotion scores (×100 for scale)",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("plot3_genre_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()



# ML CLUSTERING

print(" " + "="*60)
print("PART 2: ML Clustering")
print("="*60)

feature_cols = ["joy", "anger", "sadness", "fear",
                "disgust", "surprise", "neutral", "emotional_intensity",
                "audienceScore", "tomatoMeter"]
X        = df[feature_cols].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)


print(" Evaluating k values...")
sil_scores = []
db_scores  = []
k_range    = range(2, 8)

for k in k_range:
    mbk    = MiniBatchKMeans(n_clusters=k, random_state=42,
                              batch_size=10000, n_init=10)
    labels = mbk.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=5000))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    print(f"  k={k}: Silhouette={sil_scores[-1]:.3f}, DB={db_scores[-1]:.3f}")

best_k_auto = list(k_range)[np.argmax(sil_scores)]
best_k = 7
print(f"Mathematisch optimales k (Silhouette): {best_k_auto}")
print(f"Gewähltes k = {best_k} " f"inhaltlich besser interpretierbare Cluster")


#plot 4: Evaluation metrics
print(" Plot 4: Evaluation Metrics ")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(k_range, sil_scores, marker='o', color="#55A868",
         linewidth=2.5, markersize=8)
ax1.fill_between(k_range, sil_scores, alpha=0.1, color="#55A868")
ax1.axvline(best_k, color="red", linestyle="--", linewidth=2,
            label=f"Best k={best_k} ({max(sil_scores):.3f})")
ax1.set_title("Silhouette Score per k\n(higher = better separated clusters)",
              fontweight="bold", fontsize=12)
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Silhouette Score")
ax1.legend()
ax1.grid(alpha=0.4)

best_k_db = list(k_range)[np.argmin(db_scores)]
ax2.plot(k_range, db_scores, marker='o', color="#C44E52",
         linewidth=2.5, markersize=8)
ax2.fill_between(k_range, db_scores, alpha=0.1, color="#C44E52")
ax2.axvline(best_k, color="red", linestyle="--", linewidth=2,
            label=f"Best k={best_k} ({min(db_scores):.3f})")
ax2.set_title("Davies-Bouldin Score per k\n(lower = more compact clusters)",
              fontweight="bold", fontsize=12)
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Davies-Bouldin Score")
ax2.legend()
ax2.grid(alpha=0.4)

fig.suptitle(f"How Many Emotion Clusters Are Optimal?\n"
             f"→ Best k={best_k} based on Silhouette Score",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot4_evaluation_metrics.png", dpi=150, bbox_inches="tight")
plt.show()



print(" Training MiniBatchKMeans k={best_k} ")
mbk_final = MiniBatchKMeans(n_clusters=best_k, random_state=42,
                             batch_size=10000, n_init=10)
df["cluster"] = mbk_final.fit_predict(X_scaled) + 1
cluster_means = df.groupby("cluster")[
    ["joy","anger","sadness","fear","disgust","surprise","neutral",
     "emotional_intensity","audienceScore","tomatoMeter"]].mean()
cluster_pos = df.groupby("cluster")["scoreSentiment"].apply(
    lambda x: 100*(x=="POSITIVE").mean())

emotions_with_neutral = ["joy", "anger", "sadness", "fear",
                          "disgust", "surprise", "neutral"]

cluster_name_map = {}

for cid in cluster_means.index:
    dom_emotion = cluster_means.loc[cid, emotions_with_neutral].idxmax()
    dom_val     = cluster_means.loc[cid, emotions_with_neutral].max()
    tom = cluster_means.loc[cid, "tomatoMeter"]
    pos = cluster_pos[cid]
    ei  = cluster_means.loc[cid, "emotional_intensity"]

    reception = "Positive" if pos >= 60 else \
                "Negative" if pos < 40 else "Mixed"
    intensity = "High" if ei > 0.7 else \
                "Medium" if ei > 0.4 else "Low"

    cluster_name_map[cid] = (
        f"{dom_emotion.capitalize()} ({dom_val:.2f})\n"
        f"{reception} | {intensity} Intensity\n"
        f"Tomato: {tom:.0f}%")

    print(f"  Cluster {cid}: dominant={dom_emotion} ({dom_val:.3f}) | "
          f"tomato={tom:.0f}% | positive={pos:.0f}% | intensity={ei:.3f}")

df["cluster_name"] = df["cluster"].map(cluster_name_map)

print("Cluster name check:")
print(df["cluster_name"].value_counts())
missing = df["cluster_name"].isnull().sum()
print("Missing names: {missing}")
cluster_order = [cluster_name_map[i]
                 for i in sorted(cluster_name_map.keys(),
                                 key=lambda x: cluster_means.loc[x,"tomatoMeter"])]

cmap_colors = plt.cm.tab10(np.linspace(0, 1, best_k))
cluster_colors = [cmap_colors[i] for i in range(best_k)]

sil_final = silhouette_score(X_scaled, df["cluster"].values, sample_size=5000)
db_final  = davies_bouldin_score(X_scaled, df["cluster"].values)
print(f"Silhouette: {sil_final:.3f} | Davies-Bouldin: {db_final:.3f}")



#plot 6: Cluster profiles
print(" Plot 6: Cluster Profiles ")

aud_vals = [df[df["cluster_name"]==c]["audienceScore"].mean()
            for c in cluster_order]
tom_vals = [df[df["cluster_name"]==c]["tomatoMeter"].mean()
            for c in cluster_order]
pos_vals = [100*(df[df["cluster_name"]==c]["scoreSentiment"]=="POSITIVE").mean()
            for c in cluster_order]

xticklabels = [c.split("\n")[0] for c in cluster_order]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

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
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{val:.0f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(xticklabels, rotation=20, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    if "Sentiment" in title:
        ax.axhline(50, color="gray", linestyle="--",
                   linewidth=1.5, label="50% baseline")
        ax.legend(fontsize=9)

fig.suptitle(f"ML Cluster Profiles — Scores & Sentiment\n"
             f"Silhouette: {sil_final:.3f} | Davies-Bouldin: {db_final:.3f}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot6_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.show()


# plot 7: Box plots
print(" Plot 7: Box Plots ")

data_ei  = [df[df["cluster_name"]==c]["emotional_intensity"].values
            for c in cluster_order]
data_tom = [df[df["cluster_name"]==c]["tomatoMeter"].values
            for c in cluster_order]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, data, title, ylabel, ymax, fmt in zip(
        axes,
        [data_ei, data_tom],
        ["Emotional Intensity Distribution\nper Cluster",
         "Tomatometer Distribution\nper Cluster"],
        ["Emotional Intensity (0–1)", "Tomatometer (%)"],
        [1.1, 110],
        [".2f", ".0f"]):

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2.5),
                    flierprops=dict(marker='o', markersize=2, alpha=0.2))

    for patch, color in zip(bp["boxes"], cluster_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set_linestyle("--")

    if ymax > 2:
        ax.axhspan(0,  40,  alpha=0.05, color="#C44E52", label="Low (0–40%)")
        ax.axhspan(40, 70,  alpha=0.05, color="#F5C518", label="Medium (40–70%)")
        ax.axhspan(70, 110, alpha=0.05, color="#55A868", label="High (70–100%)")
    else:
        ax.axhspan(0,   0.4, alpha=0.05, color="#C44E52", label="Low")
        ax.axhspan(0.4, 0.7, alpha=0.05, color="#F5C518", label="Medium")
        ax.axhspan(0.7, 1.1, alpha=0.05, color="#55A868", label="High")

    for i, (d, c) in enumerate(zip(data, cluster_order)):
        median = np.median(d)
        label  = f"{median:{fmt}}" if fmt==".2f" else f"{median:.0f}%"
        ax.text(i+1, median+ymax*0.015, label,
                ha="center", fontsize=8, fontweight="bold")

    xlabels = [f"{c.split(chr(10))[0]}\n(n={(df['cluster_name']==c).sum():,})"
               for c in cluster_order]
    ax.set_xticks(range(1, len(cluster_order) + 1))
    ax.set_xticklabels(xlabels, rotation=10, ha="right",
                       fontsize=8, fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(-ymax*0.1, ymax)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Score & Intensity Distributions per Cluster\n"
             "Box = middle 50% | Line = median | Dots = outliers",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot7_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()


# mit vs ohne emotional_intensity
print("Vergleich: Mit vs Ohne emotional_intensity ")

feature_cols_with    = ["joy","anger","sadness","fear","disgust","surprise",
                        "emotional_intensity","audienceScore","tomatoMeter"]
feature_cols_without = ["joy","anger","sadness","fear","disgust","surprise",
                        "audienceScore","tomatoMeter"]

for label, cols in [("mit emotional_intensity", feature_cols_with),
                     ("ohne emotional_intensity", feature_cols_without)]:
    X_test = df[cols].values
    X_test_scaled = StandardScaler().fit_transform(X_test)
    mbk_test = MiniBatchKMeans(n_clusters=7, random_state=42,
                               batch_size=10000, n_init=10)
    labels_test = mbk_test.fit_predict(X_test_scaled)
    sil_test = silhouette_score(X_test_scaled, labels_test, sample_size=5000)
    print(f"  {label}: Silhouette = {sil_test:.3f}")


# SUMMARY

print(" " + "="*60)
print("COMPLETE CLUSTERING SUMMARY")
print("="*60)

print("PART 1: Genre based Emotion Analysis ")
print(f"Total reviews: {len(df):,}")
print(f"Top emotion per genre:")
for g in genre_list:
    dom = genre_emotion.loc[g].idxmax()
    val = genre_emotion.loc[g].max()
    tom = genre_scores[genre_scores["main_genre"]==g]["tomatoMeter"].values[0]
    pos = genre_scores[genre_scores["main_genre"]==g]["positive_pct"].values[0]
    print(f"  {g}: dominant={dom} ({val:.3f}) | tomato={tom:.0f}% | positive={pos:.0f}%")

print(" PART 2: ML Clustering")
print(f"Algorithm:       MiniBatchKMeans")
print(f"Features:        {feature_cols}")
print(f"Best k:          {best_k}")
print(f"Silhouette:      {sil_final:.3f}")
print(f"Davies-Bouldin:  {db_final:.3f}")
print(f"Cluster profiles:")
for cname in cluster_order:
    n   = (df["cluster_name"]==cname).sum()
    tom = df[df["cluster_name"]==cname]["tomatoMeter"].mean()
    pos = 100*(df[df["cluster_name"]==cname]["scoreSentiment"]=="POSITIVE").mean()
    ei  = df[df["cluster_name"]==cname]["emotional_intensity"].mean()
    print(f"  {cname.split(chr(10))[0]}: "
          f"n={n:,} | tomato={tom:.0f}% | positive={pos:.0f}% | intensity={ei:.3f}")

print(" Done! All 7 plots saved.")

print(df["cluster"].value_counts())
print(df["cluster_name"].value_counts())