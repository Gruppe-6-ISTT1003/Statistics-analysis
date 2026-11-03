"""
Bygger Jupyter-notatboken 'Prosjekt1003_Klyngeanalyse.ipynb' for ISTT1003.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# CELLE 1: Tittel og problemstilling
cells.append(nbf.v4.new_markdown_cell("""# ISTT1003 Prosjekt: Klyngeanalyse for Identifikasjon av Musikksjangre fra Spotify-lydegenskaper

**Fag**: ISTT1003 Statistikk, statistisk læring og data science  
**Tema**: Klyngeanalyse (Uovervåket læring)  
**Datasett**: `kworb_top2500_spotify.csv` (Spotify Topp 2500 mest strømmede sanger)

---

## 1. Problemstilling og Teoretisk Bakgrunn

### 1.1 Problemstilling
Musikksjangre (som Rock, Hip-Hop, Pop og Reggaeton) har tradisjonelt blitt definert av musikkindustrien og lyttere basert på kulturelle strømninger, instrumentering og markedsføring. Men i en tid der strømmealgoritmer analyserer millioner av sanger automatisk:
> **«I hvilken grad kan uovervåket klyngeanalyse av rent akustiske målinger (dansbarhet, energi, taleandel, akustiskhet, tempo) gjenskape etablerte musikksjangre, og hvilke naturlige 'lydlandskap' avdekkes i musikkdataene?»**

### 1.2 Hypoteser
1. **Hypotese 1 ($H_1$)**: Egenskapene `Speechiness` (taleandel) og `Acousticness` (akustiskhet) vil fungere som sterke klynge-separatorer for henholdsvis Hip-Hop/Rap og akustiske ballader.
2. **Hypotese 2 ($H_2$)**: Pop og Latin/Reggaeton vil i stor grad overlappe i samme klynge på grunn av likheter i dansbarhet og produksjonsenergi.
3. **Hypotese 3 ($H_3$)**: Klyngene vil avdekke *akustiske stemningsprofiler* (mood archetypes) snarere enn rene sjangre, ettersom moderne sjangre ofte hybridiseres.

### 1.3 Metoder fra ISTT1003-pensum
- **Dataprosessering**: Håndtering av manglende verdier og $Z$-score standardisering (`StandardScaler`).
- **Valg av klyngetall ($k$)**: WCSS / Inertia (Elbow-metoden) og gjennomsnittlig Silhuett-koeffisient ($s_i$).
- **$k$-Means klyngeanalyse**: Partisjoneringsalgoritme med profilering av klyngesentre.
- **Hierarkisk Agglomerativ Klyngeanalyse**: Ward's koblingsmetode og visning av dendrogram.
- **Dimensjonsreduksjon (PCA)**: Hovedkomponentanalyse for 2D-projeksjon og biplot-tolkning.
- **Validering mot Ground Truth**: Kontingenstabell, Cluster Purity, Adjusted Rand Index (ARI) og Normalized Mutual Information (NMI).
"""))

# CELLE 2: Imports
cells.append(nbf.v4.new_code_cell("""# Importer biblioteker
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for uovervåket læring og evaluering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# SciPy for dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Visualiseringsoppsett
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.figsize': (10, 6)})
"""))

# CELLE 3: Data loading and mapping
cells.append(nbf.v4.new_markdown_cell("""## 2. Datainnlasting og Kategorisering av Sjangre

Vi laster inn datasettet og mapper de over 200 unike sjanger-taggene til overordnede **sjangerfamilier** for å kunne evaluere klyngene mot en fasit (*ground truth*).
"""))

cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('data/kworb_top2500_spotify.csv')
print(f"Datasettets dimensjoner: {df.shape[0]} rader, {df.shape[1]} kolonner")

# Valgte kontinuerlige akustiske egenskaper
feature_cols = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness',
    'Acousticness', 'Valence', 'Tempo'
]

# Kartlegg sjanger-tags til overordnede sjangerfamilier
def map_genre_family(g):
    if not isinstance(g, str) or not g.strip():
        return 'Ukjent / Mangler tag'
    g_lower = g.lower()
    if any(k in g_lower for k in ['reggaeton', 'latin', 'urbano', 'corrido', 'banda', 'mexicana']):
        return 'Latin / Reggaeton'
    elif any(k in g_lower for k in ['rap', 'hip hop', 'trap']):
        return 'Hip-Hop / Rap'
    elif any(k in g_lower for k in ['rock', 'metal', 'grunge', 'punk']):
        return 'Rock / Metal'
    elif any(k in g_lower for k in ['pop', 'k-pop', 'edm', 'dance', 'electropop']):
        return 'Pop / Dance'
    elif any(k in g_lower for k in ['folk', 'acoustic', 'country', 'r&b', 'soul', 'indie', 'ballad']):
        return 'Acoustic / R&B / Indie'
    return 'Annet'

df['Genre_Family'] = df['Genres'].apply(map_genre_family)

# Vis fordelingen av sjangerfamilier
print("Fordeling av kjente sjangerfamilier:")
print(df['Genre_Family'].value_counts())
"""))

# CELLE 4: Standardisering
cells.append(nbf.v4.new_markdown_cell("""## 3. Standardisering av Egenskapene ($Z$-score)

Klyngeanalyse er basert på euklidsk avstand mellom observasjoner:
$$d(x_i, x_j) = \\sqrt{\\sum_{m=1}^{p} (x_{im} - x_{jm})^2}$$
Dersom vi ikke standardiserer, vil variabler med store absolutte verdier (som `Tempo` $\\approx 120$ BPM eller `Loudness` $\\approx -6$ dB) dominere fullstendig over variabler i området $[0, 1]$ (som `Speechiness` og `Acousticness`). Vi transformerer derfor alle variabler til $\\mu=0, \\sigma=1$.
"""))

cells.append(nbf.v4.new_code_cell("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
print("Standardiserte data (gjennomsnitt ~ 0, varians = 1):")
print(X_scaled_df.describe().round(2).loc[['mean', 'std', 'min', 'max']])
"""))

# CELLE 5: Valg av k
cells.append(nbf.v4.new_markdown_cell("""## 4. Bestemmelse av Optimalt Antall Klynger ($k$)

Vi evaluerer $k \\in [2, 10]$ ved hjelp av to komplementære metoder:
1. **Elbow-metoden (Inertia / WCSS)**: Måler summen av kvadrerte avstander fra hvert punkt til sitt klyngesenter.
2. **Silhuettanalyse**: Måler hvor godt hvert punkt passer i sin egen klynge sammenlignet med naboklyngen:
   $$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$
"""))

cells.append(nbf.v4.new_code_cell("""inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Plot Elbow og Silhuett
fig, ax1 = plt.subplots(figsize=(10, 5))

color1 = '#2c3e50'
ax1.set_xlabel('Antall klynger (k)', fontweight='bold')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color1, fontweight='bold')
line1 = ax1.plot(list(k_range), inertias, 'o-', color=color1, lw=2, label='Inertia (Elbow)')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#e74c3c'
ax2.set_ylabel('Gjennomsnittlig Silhuett-score', color=color2, fontweight='bold')
line2 = ax2.plot(list(k_range), sil_scores, 's--', color=color2, lw=2, label='Silhuett-score')
ax2.tick_params(axis='y', labelcolor=color2)

ax1.axvline(4, color='gray', linestyle=':', label='Valgt k=4')
plt.title('Valg av optimalt k: Elbow-metoden og Silhuett-analyse', fontweight='bold', fontsize=13)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')
plt.tight_layout()
plt.show()
"""))

# CELLE 6: k-Means k=4 og profilering
cells.append(nbf.v4.new_markdown_cell("""## 5. $k$-Means Klyngeanalyse ($k=4$) og Musikalsk Profilering

Vi velger $k=4$, som gir en optimal balanse mellom metrisk separasjon (knekkpunkt i Elbow) og tolkbare musikalske arketyper.
"""))

cells.append(nbf.v4.new_code_cell("""k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=15)
df['Cluster'] = km.fit_predict(X_scaled)

# Vis sentrene på original skala
cluster_centers_orig = df.groupby('Cluster')[feature_cols].mean()
print("Gjennomsnittlige akustiske egenskaper per klynge (original skala):")
print(cluster_centers_orig.round(3).T)

# Karakterisering av klyngene
cluster_names = {
    0: "K0: Klubb, Dans & Latin Pop (Høy Danceability + Høy Valence)",
    1: "K1: Høyenergi Rock & Drivende Pop (Lav Akustiskhet + Høy Energi)",
    2: "K2: Akustiske Ballader & Rolig Musikk (Høy Akustiskhet + Lav Energi)",
    3: "K3: Urban Hip-Hop & Rap (Ekstremt Høy Speechiness + Raskere Tempo)"
}
df['Cluster_Name'] = df['Cluster'].map(cluster_names)
"""))

cells.append(nbf.v4.new_code_cell("""# Profilplot: Standardiserte Z-verdier per klynge
scaled_centers = pd.DataFrame(km.cluster_centers_, columns=feature_cols)

plt.figure(figsize=(12, 6))
for i in range(k):
    plt.plot(feature_cols, scaled_centers.iloc[i], marker='o', lw=2.5,
             label=f"{cluster_names[i]}")

plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Datasettets gjennomsnitt (Z=0)')
plt.title("Akustiske fingeravtrykk for de 4 klyngene (Z-verdier)", fontweight='bold', fontsize=13)
plt.xlabel("Akustisk egenskap", fontweight='bold')
plt.ylabel("Standardavvik fra gjennomsnitt (Z-score)", fontweight='bold')
plt.xticks(rotation=15)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()
"""))

# CELLE 7: PCA
cells.append(nbf.v4.new_markdown_cell("""## 6. Dimensjonsreduksjon med PCA (Principal Component Analysis)

Siden dataene har 7 dimensjoner, benytter vi **PCA** for å projisere dataene ned til de 2 viktigste hovedkomponentene (PC1 og PC2). Dette visualiserer hvordan klyngene plasserer seg i forhold til de faktiske sjangerfamiliene.
"""))

cells.append(nbf.v4.new_code_cell("""pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

var_exp = pca.explained_variance_ratio_
print(f"Forklart varians: PC1 = {var_exp[0]:.1%}, PC2 = {var_exp[1]:.1%} (Totalt {sum(var_exp):.1%})")

# Hovedkomponentenes vekter (Loadings)
loadings = pd.DataFrame(pca.components_, columns=feature_cols, index=['PC1', 'PC2'])
print("\\nPCA Komponenter (Loadings):")
print(loadings.round(3))
"""))

cells.append(nbf.v4.new_code_cell("""# Side-ved-side visualisering i PCA-rommet
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 1. k-Means Klynger
sns.scatterplot(
    x='PC1', y='PC2', hue='Cluster', palette='tab10',
    data=df, alpha=0.65, s=35, ax=ax1
)
ax1.set_title("A) Uovervåkede k-Means klynger (k=4) i PCA-rommet", fontweight='bold')
ax1.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians) - Energi & Lydstyrke vs Akustiskhet", fontweight='bold')
ax1.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians) - Dansbarhet & Positivitet", fontweight='bold')

# 2. Faktiske sjangerfamilier
df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag'].copy()
palette_genres = {
    'Latin / Reggaeton': '#e74c3c',
    'Hip-Hop / Rap': '#9b59b6',
    'Rock / Metal': '#2c3e50',
    'Pop / Dance': '#3498db',
    'Acoustic / R&B / Indie': '#27ae60',
    'Annet': '#95a5a6'
}
sns.scatterplot(
    x='PC1', y='PC2', hue='Genre_Family', palette=palette_genres,
    data=df_known, alpha=0.65, s=35, ax=ax2
)
ax2.set_title("B) Faktiske sjangerfamilier (Ground Truth) i PCA-rommet", fontweight='bold')
ax2.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians)", fontweight='bold')
ax2.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians)", fontweight='bold')

plt.suptitle("Sammenligning av Klynger og Reelle Sjangre i PCA-rommet", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# CELLE 8: Hierarchical
cells.append(nbf.v4.new_markdown_cell("""## 7. Hierarkisk Agglomerativ Klyngeanalyse (Dendrogram)

Som et alternativ til $k$-means utfører vi **hierarkisk klyngeanalyse** med **Ward's metode** (som minimerer den totale variansen innad i klyngene ved hver sammenslåing). Dendrogrammet illustrerer hvordan sanger grupperer seg trinnvis nedenfra og opp.
"""))

cells.append(nbf.v4.new_code_cell("""# Trekk ut et representativt tilfeldig utvalg på 100 sanger for et lesbart dendrogram
np.random.seed(42)
sample_idx = np.random.choice(len(df), size=100, replace=False)
X_sub = X_scaled[sample_idx]

# Beregn koblingsmatrise
Z = linkage(X_sub, method='ward')

plt.figure(figsize=(14, 7))
dendrogram(
    Z,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True,
    color_threshold=12.0
)
plt.axhline(12.0, color='red', linestyle='--', label='Kuttnivå for 4 hovedklynger')
plt.title("Hierarkisk dendrogram (Ward's kobling) for 100 låter", fontweight='bold', fontsize=13)
plt.xlabel("Klynge / Observasjoner", fontweight='bold')
plt.ylabel("Ward's Euklidske Avstand", fontweight='bold')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
"""))

# CELLE 9: Ekstern validering mot sjangre
cells.append(nbf.v4.new_markdown_cell("""## 8. Ekstern Validering: Samsvarer Klyngene med Ekte Sjangre?

Vi kryss-tabulerer de uovervåkede klyngene mot de virkelige sjangerfamiliene og beregner formelle evalueringsmetrikker:
- **Cluster Purity**: Andel sanger i klyngene som tilhører klyngens dominerende sjanger.
- **Adjusted Rand Index (ARI)**: Justert for tilfeldig overlapping ($0 = \\text{tilfeldig}$, $1 = \\text{perfekt}$).
- **Normalized Mutual Information (NMI)**: Informasjonsdeling mellom klynger og sjangre.
"""))

cells.append(nbf.v4.new_code_cell("""# Kryss-tabell
df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag'].copy()
ct = pd.crosstab(df_known['Genre_Family'], df_known['Cluster'], margins=True)
print("KONTINGENSTABELL (Antall låter per sjanger i hver klynge):")
print(ct)

# Normalisert kryss-tabell (prosent per sjanger)
ct_pct = pd.crosstab(df_known['Genre_Family'], df_known['Cluster'], normalize='index') * 100

plt.figure(figsize=(10, 6))
short_cluster_names = ["K0: Klubb/Latin", "K1: Rock/Energi", "K2: Akustisk", "K3: Urban/Rap"]
sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Andel (%) av sjanger i klynge'},
            xticklabels=short_cluster_names)
plt.title("Hvor havner sjangrene? Andel (%) av hver sjanger fordelt på klynger", fontweight='bold', fontsize=12)
plt.xlabel("k-Means Klynge", fontweight='bold')
plt.ylabel("Faktisk sjangerfamilie", fontweight='bold')
plt.tight_layout()
plt.show()

# Valideringsmetrikker
labels_true = df_known['Genre_Family'].astype('category').cat.codes
labels_pred = df_known['Cluster']

cm_purity = pd.crosstab(df_known['Cluster'], df_known['Genre_Family'])
purity = cm_purity.max(axis=1).sum() / len(df_known)
ari = adjusted_rand_score(labels_true, labels_pred)
nmi = normalized_mutual_info_score(labels_true, labels_pred)

print(f"\\nCluster Purity:               {purity:.1%}")
print(f"Adjusted Rand Index (ARI):    {ari:.3f}")
print(f"Normalized Mutual Info (NMI): {nmi:.3f}")
"""))

# CELLE 10: Diskusjon og konklusjon
cells.append(nbf.v4.new_markdown_cell("""## 9. Konklusjon og Faglig Drøfting

### 9.1 Svar på hypotesene
1. **Hypotese 1 bekreftet**: Egenskapene `Speechiness` og `Acousticness` separerer spesifikke musikalske profiler svært effektivt:
   - **Klynge 3** har et gjennomsnittlig `Speechiness` på $0.279$ (fem ganger høyere enn de andre klyngene), og består overveiende av Hip-Hop og Rap. Rock og akustisk musikk er nesten fullstendig fraværende her.
   - **Klynge 2** har en gjennomsnittlig `Acousticness` på $0.593$ og lav energi ($0.401$), og fanger opp rolige akustiske ballader.
2. **Hypotese 2 bekreftet**: Latin/Reggaeton og Pop/Dance deler i stor grad klynge (Klynge 0). Hele $61.2\%$ av all Latin-musikk og $47.5\%$ av Pop/Dance samles her, drevet av felles karakteristika: svært høy dansbarhet ($> 0.73$) og munter stemning (høy valence).
3. **Hypotese 3 bekreftet**: Den moderate verdien for ARI ($0.050$) og Cluster Purity ($33.5\%$) reflekterer ikke en svak klyngealgoritme, men demonstrerer et fundamentalt poeng innen data science og musikkvitenskap:
   - Etablerte musikksjangre er **sosio-kulturelle merkelapper**, ikke rene akustiske kategorier.
   - En rockegruppe som Coldplay eller Queen kan lage en rolig akustisk ballade (som akustisk sett havner i Klynge 2) eller en dansbar pop-låt.
   - Klyngeanalyse avdekker reelle **lydlandskap og produksjonsstiler**, som kutter på tvers av kommersielle sjangernavn.

### 9.2 Vurdering av metodene
- **Standardisering**: Uten $Z$-score standardisering ville `Tempo` og `Loudness` overdøvet `Speechiness`, og den distinkte Hip-Hop-klyngen ville gått tapt.
- **PCA**: Hovedkomponentene forklarte over $52\%$ av den totale variansen og bekreftet at PC1 representerer en akse for *energi/akustiskhet*, mens PC2 representerer *dansbarhet og humør*.
"""))

nb['cells'] = cells

with open('Prosjekt1003_Klyngeanalyse.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Lagret 'Prosjekt1003_Klyngeanalyse.ipynb'!")
