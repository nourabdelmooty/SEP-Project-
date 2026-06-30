# SEP-Projekt: Analyse von Rotten-Tomatoes-Filmkritiken
Thema: ML zur Analyse von Filmrezensionen 
anhand emotionaler Profile


## Projektübersicht

Dieses Projekt untersucht Filmkritiken aus Rotten Tomatoes mithilfe von Methoden des Machine Learnings.

Der hier dokumentierte Teil des Projekts umfasst

* die Datenaufbereitung,
* die Klassifikation der Emotionsintensität von Rezensionen sowie
* die Erstellung eines Datensatzes auf Filmebene.

Die anschließend durchgeführte Regressions-Analyse basiert auf dem in Notebook 03 erzeugten Filmdatensatz. Die Clustering-Analyse verwendet separat merged_clean.csv und classification_sample_100k_with_emotions.csv.
## Datenquelle

Die verwendeten Datensätze stammen aus Kaggle:

**Clapper – Massive Rotten Tomatoes Movies and Reviews**

https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews

Verwendete Dateien:

* rotten_tomatoes_movie_reviews.csv
* rotten_tomatoes_movies.csv

## Projektstruktur

```text
notebooks/
├── 01_data_understanding.ipynb
├── 02_classification_emotion_intensity_updated.ipynb
└── 03_film_level_updated.ipynb
└── 04_regression.ipynb
└── 05_clustering.ipynb
```

## Beschreibung der Notebooks

### 01_data_understanding.ipynb

Untersuchung und Bereinigung der ursprünglichen Rotten-Tomatoes-Daten.

Inhalte

* Laden der Originaldatensätze
* Untersuchung der Datenqualität
* Analyse fehlender Werte
* Sprachfilterung
* Entfernen von Duplikaten
* Entfernen sehr kurzer Rezensionen
* Berechnung der Letter Ratio
* Zusammenführung von Film- und Rezensionsdaten für die explorative Analyse

Erzeugte Dateien

* reviews_clean.csv
* merged_clean.csv (nur für die explorative Datenanalyse)

### 02_classification_emotion_intensity_updated.ipynb

Klassifikation der Emotionsintensität von Filmkritiken.

Inhalte

* Berechnung von Emotionsscores
* Erstellung der Zielvariable
* Explorative Datenanalyse
* TF-IDF-Vektorisierung
* Training mehrerer Klassifikationsmodelle
* Modellevaluation
* Erzeugung des finalen Datensatzes

Erzeugte Dateien

Zwischenergebnisse

* classification_sample_50k_with_emotions.csv
* classification_sample_100k_with_emotions.csv
* classification_sample_500k_with_emotions.csv
* classification_full_with_emotions.csv

Vorhersagen

* classification_predictions_50k.csv
* classification_predictions_100k.csv
* classification_predictions_500k.csv
* classification_predictions_full.csv

Finaler Datensatz

* reviews_with_emotions_and_intensity_classes.csv

### 03_film_level_updated.ipynb

Aggregation der Rezensionen auf Filmebene.

Inhalte

* Aggregation aller Rezensionen eines Films
* Berechnung aggregierter Emotionsmerkmale
* Erstellung eines Filmdatensatzes

Erzeugte Dateien

* film_level_full.csv
* film_level_clean.csv

### 04_regression.ipynb
Aufbau und Evaluation von Regressionsmodellen zur Vorhersage von Filmbewertungen auf Basis von Kritiker-Emotionen und Filmmerkmalen.

Inhalte

*  Laden des aggregierten Film-Datensatzes `film_level_dataset_reliable.csv`
*  Explorative Analyse der Zielvariable `audienceScore`
*  Behandlung fehlender Werte
*  One-Hot-Encoding kategorialer Merkmale
*  Analyse von Multikollinearität zwischen Emotionsmerkmalen
* Training und Evaluation von Linearer Regression und Random Forest
* Vergleich verschiedener Modellversionen mit und ohne `tomatoMeter`
* Konstruktion der alternativen Zielvariable `meinungsverschiedenheit = audienceScore - tomatoMeter`
* Ergänzung zusätzlicher Merkmale aus `merged_clean.csv`
* Wechsel der finalen Zielvariable zu `tomatoMeter`, da die Emotionsdaten aus Kritikerrezensionen stammen
* Extraktion zusätzlicher Filmmerkmale wie Kasseneinnahmen, Altersfreigabe, Kinostart-Datum und Genre-Informationen
* Finales Modelltraining zur Vorhersage von `tomatoMeter`
* Prüfung von Overfitting anhand von Train- und Test-R²
* Analyse von Koeffizienten und Feature Importance

Verwendete Dateien

* film_level_dataset_reliable.csv (Eingabe)
* merged_clean.csv (Eingabe, für zusätzliche Merkmale)

### 05_clustering.ipynb

Clustering-Analyse der Filmrezensionen anhand von Bewertungsmustern und emotionalen Profilen.

Inhalte

* Deskriptive Genre-Analyse (Kritiker- vs. Publikumsbewertung)
* ML-Clustering anhand von Bewertungsmustern (Modell 1)
* Deskriptive Genre-Emotionsanalyse
* ML-Clustering anhand von Emotionsprofilen (Modell 2)
* Evaluation beider Modelle mit Silhouette Score und Davies-Bouldin Score
* Vergleichstest mit und ohne emotional_intensity als Feature

Verwendete Dateien

* merged_clean.csv (Eingabe, Modell 1)
* classification_sample_100k_with_emotions.csv (Eingabe, Modell 2, gemergt mit merged_clean.csv)

Erzeugte Dateien

Modell 1 — Rezeptionsbasiertes Clustering

* plot1_scores_per_genre.png
* plot2_positive_negative.png
* plot3_genre_heatmap.png
* plot4_evaluation.png
* plot5_cluster_profiles.png
* plot6_scatter_comparison.png

Modell 2 — Emotionsbasiertes Clustering

* plot1_emotions_per_genre.png
* plot2_dominant_genre_per_emotion.png
* plot3_genre_heatmap.png
* plot4_evaluation_metrics.png
* plot5_cluster_profiles.png
* plot6_boxplots.png
## Verarbeitungsablauf
```text
Kaggle

rotten_tomatoes_movie_reviews.csv
rotten_tomatoes_movies.csv
            │
            ▼
01_data_understanding.ipynb
            │
            ├── reviews_clean.csv
            └── merged_clean.csv (nur für EDA und Clustering)
                    │
                    ▼
02_classification_emotion_intensity_updated.ipynb
            │
            ▼
reviews_with_emotions_and_intensity_classes.csv
            │
            ▼
03_film_level_updated.ipynb
            │
            ▼
film_level_clean.csv
            │
            ▼
04_regression.ipynb
           
05_clustering.ipynb ( verwendet merged_clean.csv und
classification_sample_100k_with_emotions.csv)
```

## Datensätze

| Datei                                             | Beschreibung | Erstellt durch |
|---------------------------------------------------|--------------|----------------|
| `reviews_clean.csv`                               | Bereinigte englischsprachige Filmrezensionen. Dieser Datensatz dient als Ausgangspunkt für die Klassifikation. | 01_data_understanding |
| `reviews_with_emotions_and_intensity_classes.csv` | Rezensionen mit berechneten Emotionsscores sowie der finalen Zielvariable (`low`, `medium`, `high`). Dieser Datensatz wird für die Klassifikation erstellt und dient anschließend als Grundlage für die Aggregation auf Filmebene. | 02_classification_emotion_intensity_updated |
| `film_level_clean.csv`                            | Aggregierter Datensatz auf Filmebene. Die aus den Rezensionen berechneten Emotionsmerkmale werden je Film zusammengefasst und mit den Filminformationen kombiniert. Dieser Datensatz bildet die Grundlage für die Regressions- und Clustering-Analysen. | 03_film_level_updated |
| `classification_sample_100k_with_emotions.csv`    |  Stichprobe für Experimente während der Entwicklung der Klassifikation.              |   02_classification_emotion_intensity_updated   |
| `film_level_dataset_full.csv`                     | Aggregierter Filmdatensatz aus einer früheren Projektphase; wird von weiteren Projektteilen verwendet.             |                       |
| `film_level_dataset_reliable.csv`                 |  Bereinigte Variante des aggregierten Filmdatensatzes aus der bisherigen Projektpipeline.            |                       |
| `merged_clean.csv` (gefiltert)                    | Wird zusätzlich im Clustering-Notebook verwendet — auf die sechs häufigsten Genres reduziert (Comedy, Drama, Documentary, Mystery & Thriller, Action, Horror). | 01_data_understanding, weiterverarbeitet in 05_clustering |



## Hinweise
* merged_clean.csv dient ausschließlich der explorativen Datenanalyse.
* Die Klassifikation arbeitet ausschließlich mit reviews_clean.csv.
* Die Dateien film_level_dataset_full.csv und film_level_dataset_reliable.csv stammen aus einer früheren Projektphase und bleiben aus Gründen der Nachvollziehbarkeit sowie zur Kompatibilität mit bereits entwickelten Notebooks im Repository erhalten. Für die finale Pipeline wird film_level_clean.csv verwendet.
* Die Regressionsmodell verwendet den in Notebook 03 erzeugten Datensatz film_level_clean.csv.
* Das Clustering-Notebook arbeitet abweichend von Regression und Klassifikation auf Review-Ebene statt Film-Ebene und verwendet daher `merged_clean.csv` sowie `classification_sample_100k_with_emotions.csv` direkt anstelle von `film_level_clean.csv`. Modell 1 nutzt ausschließlich Bewertungswerte (audienceScore, tomatoMeter). Genre dient nur der nachträglichen Interpretation. Modell 2 ergänzt zusätzlich Emotionsmerkmale.

## Voraussetzungen

Python 3.13

Benötigte Bibliotheken:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* nltk
* langdetect
* transformers
* torch
