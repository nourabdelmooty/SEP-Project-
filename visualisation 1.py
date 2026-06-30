import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

merged = pd.read_csv('Dataset/merged_clean.csv')

#plot 1 : tomatometer score distribution
plt.figure(figsize=(10, 5))
merged["tomatoMeter"].dropna().hist(bins=50, color="tomato")
plt.title("Tomatometer Score Distribution")
plt.xlabel("Tomatometer Score (%)")
plt.ylabel("Number of Reviews")
plt.grid(axis="y")
plt.show()

#plot 2: fresh vs rotten
merged["freshness"] = merged["tomatoMeter"].apply(
    lambda x: "Fresh" if x >= 60 else "Rotten" if pd.notna(x) else None)

merged["freshness"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Fresh vs Rotten")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()

#plot 3: critic score vs audience score distribution
fig, ax = plt.subplots(figsize=(10, 6))

merged["tomatoMeter"].dropna().plot(kind="hist", bins=50, alpha=0.6,
                                     color="tomato", label="Critic Score (Tomatometer)")
merged["audienceScore"].dropna().plot(kind="hist", bins=50, alpha=0.6,
                                       color="steelblue", label="Audience Score")

plt.axvline(merged["tomatoMeter"].mean(), color="red", linestyle="--",
            linewidth=2, label=f"Critic Mean: {merged['tomatoMeter'].mean():.1f}%")
plt.axvline(merged["audienceScore"].mean(), color="blue", linestyle="--",
            linewidth=2, label=f"Audience Mean: {merged['audienceScore'].mean():.1f}%")

plt.title("Do Critics and Audiences Rate Films Differently?")
plt.xlabel("Score (%)")
plt.ylabel("Number of Films")
plt.legend()
plt.grid(axis="y")
plt.show()


#plot 4: top 10 most reviewed genres
genre_counts = (merged["genre"].dropna()
                .str.split(",")
                .explode()
                .str.strip()
                .value_counts()
                .head(10))

genre_counts.plot(kind="barh", color="tomato")
plt.title("Top 10 Most Common Genres")
plt.xlabel("Number of Reviews")
plt.grid(axis="x")
plt.show()

#plot 5: most reviewed films
top_films = (merged.groupby("title")["reviewId"]
             .count()
             .sort_values()
             .tail(10))

plt.figure(figsize=(12, 6))
top_films.plot(kind="barh", color="tomato")
plt.title("Top 10 Most Reviewed Films")
plt.xlabel("Number of Reviews")
plt.grid(axis="x")
plt.tight_layout()
plt.subplots_adjust(left=0.3)
plt.show()


#plot 6 : fresh vs rotten count over genres
merged["freshness"] = merged["tomatoMeter"].apply(
    lambda x: "Fresh" if x >= 60 else "Rotten" if pd.notna(x) else None)

genre_fresh = merged[["genre", "freshness"]].dropna()
genre_fresh["genre"] = genre_fresh["genre"].str.split(",").str[0].str.strip()

top_genres = genre_fresh["genre"].value_counts().head(8).index
genre_fresh = genre_fresh[genre_fresh["genre"].isin(top_genres)]

genre_counts = (genre_fresh.groupby(["genre", "freshness"])
                .size()
                .unstack(fill_value=0))

genre_counts.plot(kind="bar", stacked=True,
                  color=["tomato", "steelblue"],
                  figsize=(12, 6), edgecolor="black", linewidth=0.5)

plt.title("Fresh vs Rotten Count per Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Films")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Freshness")
plt.grid(axis="y")
plt.tight_layout()
plt.show()



#plot 7: review count vs score
review_counts = merged.groupby("title").agg(
    review_count=("reviewId", "count"),
    avg_score=("tomatoMeter", "mean")
).dropna()

bins = [0, 10, 50, 100, 500, 1000, 99999]
labels = ["1-10", "11-50", "51-100", "101-500", "501-1000", "1000+"]
review_counts["review_bin"] = pd.cut(review_counts["review_count"],
                                      bins=bins, labels=labels)

avg_per_bin = review_counts.groupby("review_bin")["avg_score"].mean()

plt.figure(figsize=(10, 6))
bars = plt.bar(avg_per_bin.index, avg_per_bin.values,
               color="tomato", edgecolor="black", linewidth=0.5, width=0.5)

for bar, val in zip(bars, avg_per_bin.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=11, fontweight="bold")

plt.title("Does Popularity Affect Critic Scores?")
plt.xlabel("Number of Reviews per Film")
plt.ylabel("Average Tomatometer Score (%)")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.tight_layout()
plt.show()
