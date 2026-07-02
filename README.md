# SEP-Projekt: Analyse von Rotten-Tomatoes-Filmkritiken
Thema: ML zur Analyse von Filmrezensionen 
anhand emotionaler Profile


## Projektübersicht
Dieses Projekt untersucht Filmrezensionen von Rotten Tomatoes mithilfe von Methoden des Machine Learnings.

Ziel ist es, emotionale Informationen aus Rezensionen zu extrahieren und diese für verschiedene Analyseaufgaben auf Review- und Filmebene zu nutzen.

Die Analysen bauen inhaltlich aufeinander auf:

1. **Datenaufbereitung** – Bereinigung und Vorbereitung der Rohdaten.
2. **Klassifikation** – Vorhersage der emotionalen Intensität einzelner Rezensionen.
3. **Aggregation** – Zusammenführung der Emotionen auf Filmebene.
4. **Regression** – Vorhersage des TomatoMeter eines Films anhand aggregierter Emotionen und weiterer Filmmerkmale.
5. **Clustering** – Identifikation ähnlicher Bewertungs- und Emotionsprofile.

Weitere Informationen zu den einzelnen Schritten, den verwendeten Datensätzen und der Datenpipeline finden sich in den folgenden Abschnitten.

## Repository-Struktur

```text
SEP-Project/
│
├── Dataset/
│   ├── film_level_clean.csv
│   └── classification_sample_100k_with_emotions.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_classification_emotion_intensity_updated.ipynb
│   ├── 03_film_level_updated.ipynb
│   ├── 04_regression.ipynb
│   └── 05_clustering.ipynb
│
├── README.md
└── LICENSE
```

## Datenquellen
### Rotten Tomatoes

Kaggle:
**Clapper – Massive Rotten Tomatoes Movies and Reviews**

https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews

Verwendete Rohdaten:
* rotten_tomatoes_movie_reviews.csv
* rotten_tomatoes_movies.csv

### Emotionsmodell (Emotionserkennung)

Hugging-Face-Modell:
* j-hartmann/emotion-english-distilroberta-base

## Datenpipeline
```text
Kaggle

rotten_tomatoes_movie_reviews.csv
rotten_tomatoes_movies.csv
                │
                ▼
      01_data_understanding.ipynb
                │
      ┌─────────┴─────────┐
      ▼                   ▼
reviews_clean.csv   merged_clean.csv
+ Emotionsmodell            │  
      │                     │
      ▼                     ├──────────────► 04_regression.ipynb
02_classification_          │               (zusätzliche Filmmerkmale)
emotion_intensity_          │
updated.ipynb               └──────────────► 05_clustering.ipynb
      │                                     (Modell 1,2)
      ▼
classification_sample_100k_with_emotions.csv
      │
      ├──────────────► 05_clustering.ipynb
      │               (Modell 2)
      ▼
03_film_level_updated.ipynb
      │
      ▼
film_level_clean.csv
      │
      ▼
04_regression.ipynb
(Hauptdatensatz)
```

## Verfügbare und nicht enthaltene Datensätze

### Im Repository enthalten

Die folgenden Datensätze werden im Repository gespeichert:

| Datei | Verwendung |
|--------|------------|
| film_level_clean.csv | Eingabe für die Regression |
| classification_sample_100k_with_emotions.csv | Eingabe für das emotionsbasierte Clustering (Modell 2) |

### Nicht im Repository enthalten

Die folgenden Dateien werden **nicht** im Repository gespeichert:

| Datei | Grund                                         |
|--------|-----------------------------------------------|
| rotten_tomatoes_movie_reviews.csv | ursprünglicher Kaggle-Datensatz  |
| rotten_tomatoes_movies.csv | ursprünglicher Kaggle-Datensatz   |
| reviews_clean.csv | automatisch in Notebook 01 erzeugt  (Dateigröße) |
| merged_clean.csv | automatisch in Notebook 01 erzeugt  (Dateigröße) |
| reviews_with_emotions_and_intensity_classes.csv | automatisch in Notebook 02 erzeugt  (Dateigröße) |

Die ursprünglichen Kaggle-Datensätze können jederzeit über den oben angegebenen Link heruntergeladen werden.

Die automatisch erzeugten Zwischendatensätze werden beim erneuten Ausführen der Pipeline erstellt und müssen daher nicht im Repository gespeichert werden.

## Beschreibung der Notebooks

### 01_data_understanding.ipynb

Untersuchung und Bereinigung der ursprünglichen Rotten-Tomatoes-Daten.

**Inhalte**:

* Laden der Originaldatensätze
* Untersuchung der Datenqualität
* Analyse fehlender Werte
* Sprachfilterung
* Entfernen von Duplikaten
* Entfernen sehr kurzer Rezensionen
* Berechnung der Letter Ratio
* Zusammenführung von Film- und Rezensionsdaten für die explorative Analyse

**Erzeugte Dateien**:

* reviews_clean.csv
* merged_clean.csv

### 02_classification_emotion_intensity_updated.ipynb

Klassifikation der Emotionsintensität von Filmkritiken.

**Inhalte**:

* Berechnung von Emotionsscores
* Erstellung der Zielvariable
* Explorative Datenanalyse
* TF-IDF-Vektorisierung
* Training mehrerer Klassifikationsmodelle
* Modellevaluation
* Erzeugung des finalen Datensatzes

**Erzeugte Dateien**:

Zwischenergebnisse:

* verschiedene Datensätze mit berechneten Emotionsscores und Vorhersagen

Finaler Datensatz:

* reviews_with_emotions_and_intensity_classes.csv
* classification_sample_100k_with_emotions.csv

### 03_film_level_updated.ipynb

Aggregation der Rezensionen auf Filmebene.

**Inhalte**:

* Aggregation aller Rezensionen eines Films
* Berechnung aggregierter Emotionsmerkmale
* Erstellung eines Filmdatensatzes

**Erzeugte Dateien**:

* film_level_clean.csv

### 04_regression.ipynb
Aufbau und Evaluation von Regressionsmodellen zur Vorhersage von Filmbewertungen auf Basis von Kritiker-Emotionen und Filmmerkmalen.

**Inhalte**:

*  Laden des aggregierten Film-Datensatzes `film_level_clean.csv`
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

**Verwendete Dateien**:

* film_level_clean.csv (Eingabe)
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
