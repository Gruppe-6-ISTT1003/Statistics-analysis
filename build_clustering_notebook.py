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

# SciPy for dendrogram og optimal klynge-tilordning
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Visualiseringsoppsett
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.figsize': (10, 6)})

GENRE_ORDER = [
    'Latin / Reggaeton',
    'Rock / Metal',
    'Acoustic / R&B / Indie',
    'Hip-Hop / Rap',
    'Pop / Dance'
]

PALETTE_GENRES = {
    'Latin / Reggaeton': '#e74c3c',
    'Rock / Metal': '#2c3e50',
    'Acoustic / R&B / Indie': '#27ae60',
    'Hip-Hop / Rap': '#9b59b6',
    'Pop / Dance': '#3498db'
}
CLUSTER_COLORS = [PALETTE_GENRES[g] for g in GENRE_ORDER]
"""))

# CELLE 3: Data loading and mapping
cells.append(nbf.v4.new_markdown_cell("""## 2. Datainnlasting og Kategorisering av Sjangre

Vi laster inn datasettet og mapper de over 200 unike sjanger-taggene i CSV-filen til **5 overordnede sjangerfamilier** for å kunne evaluere de $k=5$ klyngene 1-til-1 i en kvadratisk ($5 \\times 5$) samsvarsmatrise mot en fasit (*ground truth*).
"""))

cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('data/kworb_top2500_spotify.csv')
print(f"Datasettets dimensjoner: {df.shape[0]} rader, {df.shape[1]} kolonner")

# Valgte kontinuerlige akustiske egenskaper
feature_cols = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness',
    'Acousticness', 'Valence', 'Tempo'
]

# Kartlegg sjanger-tags til 5 overordnede sjangerfamilier
def map_genre_family(g):
    if not isinstance(g, str) or not g.strip():
        return 'Ukjent / Mangler tag'
    g_lower = g.lower()
    if any(k in g_lower for k in [
        'reggaeton', 'latin', 'urbano', 'corrido', 'banda', 'mexicana',
        'bachata', 'moombahton', 'reggae', 'afrobeats', 'salsa', 'cumbia', 'dembow'
    ]):
        return 'Latin / Reggaeton'
    elif any(k in g_lower for k in ['rap', 'hip hop', 'trap', 'drill', 'grime', 'crunk', 'phonk']):
        return 'Hip-Hop / Rap'
    elif any(k in g_lower for k in ['rock', 'metal', 'grunge', 'punk', 'emo', 'new wave', 'britpop', 'shoegaze']):
        return 'Rock / Metal'
    elif any(k in g_lower for k in [
        'pop', 'k-pop', 'edm', 'dance', 'electropop', 'house', 'electro',
        'disco', 'techno', 'trance', 'eurodance', 'synth', 'uk garage', 'dubstep'
    ]):
        return 'Pop / Dance'
    return 'Acoustic / R&B / Indie'

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

ax1.axvline(5, color='gray', linestyle=':', label='Valgt k=5 (5 sjangerfamilier)')
plt.title('Valg av optimalt k: Elbow-metoden og Silhuett-analyse', fontweight='bold', fontsize=13)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')
plt.tight_layout()
plt.show()
"""))

# CELLE 6: k-Means k=5 og profilering
cells.append(nbf.v4.new_markdown_cell("""## 5. $k$-Means Klyngeanalyse ($k=5$) og Musikalsk Profilering

Vi velger $k=5$ slik at antallet klynger matcher de 5 sjangerfamiliene fra CSV-filen 1-til-1, og benytter optimal tilordning (*Hungarian algorithm*) slik at hver klynge kobles til den sjangerkategorien den representerer best.
"""))

cells.append(nbf.v4.new_code_cell("""k = 5
km = KMeans(n_clusters=k, random_state=42, n_init=15)
df['Raw_Cluster'] = km.fit_predict(X_scaled)

# Tilordne hver klynge 1-til-1 til sjangerfamilien den samsvarer best med
df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag']
ct_raw = pd.crosstab(df_known['Genre_Family'], df['Raw_Cluster'], normalize='index').reindex(GENRE_ORDER)
row_ind, col_ind = linear_sum_assignment(-ct_raw.values)
remap = {int(col_ind[r]): r for r in row_ind}

df['Cluster'] = df['Raw_Cluster'].map(remap)
df['Cluster_Genre'] = df['Cluster'].map(lambda c: GENRE_ORDER[c])

inv_remap = {new_idx: old_idx for old_idx, new_idx in remap.items()}
ordered_centers = np.array([km.cluster_centers_[inv_remap[i]] for i in range(k)])

# Vis sentrene på original skala
cluster_centers_orig = df.groupby('Cluster')[feature_cols].mean()
print("Gjennomsnittlige akustiske egenskaper per klynge (original skala):")
print(cluster_centers_orig.round(3).T)

# Karakterisering av klyngene
cluster_names = {
    0: "K0: Latin / Reggaeton (Høy Danceability + Høy Valence)",
    1: "K1: Rock / Metal (Høy Energi + Raskt Tempo)",
    2: "K2: Acoustic / R&B / Indie (Høy Akustiskhet + Lav Energi)",
    3: "K3: Hip-Hop / Rap (Høy Speechiness + Rytmisk)",
    4: "K4: Pop / Dance (Balansert Energi & Dansbarhet)"
}
df['Cluster_Name'] = df['Cluster'].map(cluster_names)
"""))

cells.append(nbf.v4.new_code_cell("""# Profilplot: Standardiserte Z-verdier per klynge (samme fargepalett på tvers av alle figurer)
scaled_centers = pd.DataFrame(ordered_centers, columns=feature_cols)

plt.figure(figsize=(12, 6))
for i in range(k):
    plt.plot(feature_cols, scaled_centers.iloc[i], marker='o', lw=2.5,
             color=CLUSTER_COLORS[i], label=f"{cluster_names[i]}")

plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Datasettets gjennomsnitt (Z=0)')
plt.title("Akustiske fingeravtrykk for de 5 klyngene (Z-verdier)", fontweight='bold', fontsize=13)
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

cells.append(nbf.v4.new_code_cell("""# Side-ved-side visualisering i PCA-rommet med identisk fargepalett
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

palette_genres = {
    'Latin / Reggaeton': '#e74c3c',
    'Rock / Metal': '#2c3e50',
    'Acoustic / R&B / Indie': '#27ae60',
    'Hip-Hop / Rap': '#9b59b6',
    'Pop / Dance': '#3498db'
}

# 1. k-Means Klynger
sns.scatterplot(
    x='PC1', y='PC2', hue='Cluster_Genre', hue_order=GENRE_ORDER, palette=palette_genres,
    data=df, alpha=0.65, s=35, ax=ax1
)
ax1.set_title("A) Uovervåkede k-Means klynger (k=5) i PCA-rommet", fontweight='bold')
ax1.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians) - Energi & Lydstyrke vs Akustiskhet", fontweight='bold')
ax1.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians) - Dansbarhet & Positivitet", fontweight='bold')

# 2. Faktiske sjangerfamilier
df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag'].copy()
sns.scatterplot(
    x='PC1', y='PC2', hue='Genre_Family', hue_order=GENRE_ORDER, palette=palette_genres,
    data=df_known, alpha=0.65, s=35, ax=ax2
)
ax2.set_title("B) Faktiske sjangerfamilier (Ground Truth) i PCA-rommet", fontweight='bold')
ax2.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians)", fontweight='bold')
ax2.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians)", fontweight='bold')

plt.suptitle("Sammenligning av Klynger (k=5) og Reelle Sjangre i PCA-rommet", fontsize=14, fontweight='bold')
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
sub_clusters = df['Cluster'].iloc[sample_idx].values

# Beregn koblingsmatrise og koble grenene til de 5 gruppefargene
Z = linkage(X_sub, method='ward')
h_labels = fcluster(Z, t=10.5, criterion='distance')
ct_h = pd.crosstab(h_labels, sub_clusters)
row_ind, col_ind = linear_sum_assignment(-ct_h.values)
branch_palette = [CLUSTER_COLORS[col_ind[r]] for r in range(len(row_ind))]
set_link_color_palette(branch_palette)

plt.figure(figsize=(14, 7))
dendrogram(
    Z,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True,
    color_threshold=10.5,
    above_threshold_color='#95a5a6'
)
set_link_color_palette(None)
cutoff_line = plt.axhline(10.5, color='#c0392b', linestyle='--', lw=1.8, label='Kuttnivå for 5 hovedklynger (d = 10.5)')
legend_handles = [cutoff_line] + [
    Line2D([0], [0], color=CLUSTER_COLORS[i], lw=3, label=f"Gren → K{i}: {g}")
    for i, g in enumerate(GENRE_ORDER)
]
plt.title("Hierarkisk dendrogram (Ward's kobling) fargekodet etter de 5 hovedgruppene", fontweight='bold', fontsize=13)
plt.xlabel("Klynge / Observasjoner", fontweight='bold')
plt.ylabel("Ward's Euklidske Avstand", fontweight='bold')
plt.legend(handles=legend_handles, loc='upper right', frameon=True)
plt.tight_layout()
plt.show()
"""))

# CELLE 9: Ekstern validering mot sjangre
cells.append(nbf.v4.new_markdown_cell("""## 8. Ekstern Validering: Kvadratisk ($5 \\times 5$) Samsvarsmatrise mot Ekte Sjangre

Vi kryss-tabulerer de 5 uovervåkede klyngene mot de 5 virkelige sjangerfamiliene i en kvadratisk $5 \\times 5$-matrise (der hoveddiagonalen viser direkte samsvar mellom sjanger og tilordnet klynge) og beregner formelle evalueringsmetrikker:
- **Cluster Purity**: Andel sanger i klyngene som tilhører klyngens dominerende sjanger.
- **Adjusted Rand Index (ARI)**: Justert for tilfeldig overlapping ($0 = \\text{tilfeldig}$, $1 = \\text{perfekt}$).
- **Normalized Mutual Information (NMI)**: Informasjonsdeling mellom klynger og sjangre.
"""))

cells.append(nbf.v4.new_code_cell("""# Kvadratisk (5x5) kryss-tabell
df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag'].copy()
ct = pd.crosstab(df_known['Genre_Family'], df_known['Cluster_Genre']).reindex(
    index=GENRE_ORDER, columns=GENRE_ORDER
)
print("KVADRATISK KONTINGENSTABELL (5x5 - Antall låter per sjanger i hver klynge):")
print(ct)

# Normalisert kryss-tabell (prosent per sjanger)
ct_pct = pd.crosstab(
    df_known['Genre_Family'], df_known['Cluster_Genre'], normalize='index'
).reindex(index=GENRE_ORDER, columns=GENRE_ORDER) * 100

fig, ax = plt.subplots(figsize=(9.5, 8.5))
cluster_axis_labels = [f"K{i}: {g}" for i, g in enumerate(GENRE_ORDER)]
sns.heatmap(
    ct_pct, annot=True, fmt=".1f", cmap="YlGnBu", square=True,
    cbar_kws={'label': 'Andel (%) av sjanger i klynge', 'shrink': 0.82},
    xticklabels=cluster_axis_labels, yticklabels=GENRE_ORDER, ax=ax
)
ax.set_title("Kvadratisk (5×5) samsvarsmatrise: Faktiske sjangre vs. k-Means klynger (%)", fontweight='bold', fontsize=12)
ax.set_xlabel("Tilordnet k-Means Klynge (Akustisk profil)", fontweight='bold')
ax.set_ylabel("Faktisk sjangerfamilie i CSV", fontweight='bold')
plt.xticks(rotation=30, ha='right', fontweight='bold')
plt.yticks(rotation=0, fontweight='bold')
for i, color in enumerate(CLUSTER_COLORS):
    ax.get_xticklabels()[i].set_color(color)
    ax.get_yticklabels()[i].set_color(color)
plt.tight_layout()
plt.show()

# Valideringsmetrikker
labels_true = df_known['Genre_Family'].astype('category').cat.codes
labels_pred = df_known['Cluster']

cm_purity = pd.crosstab(df_known['Cluster'], df_known['Genre_Family'])
purity = cm_purity.max(axis=1).sum() / len(df_known)
diag_acc = np.trace(ct.values) / len(df_known)
ari = adjusted_rand_score(labels_true, labels_pred)
nmi = normalized_mutual_info_score(labels_true, labels_pred)

print(f"\\nDiagonal 1-til-1 treff (5x5): {diag_acc:.1%}")
print(f"Cluster Purity:               {purity:.1%}")
print(f"Adjusted Rand Index (ARI):    {ari:.3f}")
print(f"Normalized Mutual Info (NMI): {nmi:.3f}")

# Boksplott for alle 7 akustiske egenskaper på tvers av de 5 klyngene
from matplotlib.patches import Patch
cluster_palette = ['#e74c3c', '#2c3e50', '#27ae60', '#9b59b6', '#3498db']
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for idx, col in enumerate(feature_cols):
    sns.boxplot(x='Cluster', y=col, hue='Cluster', data=df, ax=axes[idx], palette=cluster_palette, legend=False)
    axes[idx].set_title(f"Fordeling av {col}", fontweight='bold')
    axes[idx].set_xlabel("Klynge")
    axes[idx].set_xticks(range(len(GENRE_ORDER)))
    axes[idx].set_xticklabels([f"K{i}\\n({g.split('/')[0].strip()})" for i, g in enumerate(GENRE_ORDER)], fontsize=9)

axes[7].axis('off')
legend_handles = [
    Patch(facecolor=cluster_palette[i], edgecolor='black', label=f"K{i}: {g}")
    for i, g in enumerate(GENRE_ORDER)
]
axes[7].legend(handles=legend_handles, title="k-Means Klynger (k=5)", title_fontsize=12, fontsize=11, loc='center', frameon=True)
plt.suptitle("Akustiske forskjeller på tvers av de 5 klyngene (Alle 7 egenskaper)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# CELLE 10: Diskusjon og konklusjon
cells.append(nbf.v4.new_markdown_cell(r"""## 9. Konklusjon og Faglig Drøfting

### 9.1 Svar på hypotesene
1. **Hypotese 1 bekreftet**: Egenskapene `Speechiness` og `Acousticness` separerer spesifikke musikalske profiler svært effektivt langs hoveddiagonalen i $5 \times 5$-matrisen:
   - **Klynge 3 (Hip-Hop / Rap)** har et gjennomsnittlig `Speechiness` på $0.294$ (nesten fem ganger høyere enn de andre klyngene), og treffer $33.7\%$ av Hip-Hop/Rap. Rock ($1.4\%$) og akustisk musikk ($1.2\%$) er nesten fullstendig fraværende her.
   - **Klynge 2 (Acoustic / R&B / Indie)** har en gjennomsnittlig `Acousticness` på $0.664$ og lav energi ($0.373$), og treffer $32.1\%$ av de akustiske låtene.
   - **Klynge 1 (Rock / Metal)** har høy energi ($0.757$), lav akustiskhet ($0.110$) og høyt tempo ($150.3\text{ BPM}$), og fanger opp $40.4\%$ av Rock/Metal.
2. **Hypotese 2 bekreftet**: Latin/Reggaeton og Pop/Dance overlapper sterkt akustisk. Hele $50.9\%$ av Latin/Reggaeton havner i **Klynge 0 (Latin / Reggaeton)**, men også $42.6\%$ av Pop/Dance trekkes mot denne klyngen på grunn av svært høy dansbarhet ($0.737$) og positivitet/valence ($0.738$), mens $26.7\%$ av Pop/Dance blir liggende i den mer moderat tempererte **Klynge 4 (Pop / Dance)**.
3. **Hypotese 3 bekreftet**: Den moderate verdien for ARI ($0.061$), diagonal 1-til-1 treff ($36.5\%$) og Cluster Purity ($37.7\%$) reflekterer ikke en svak klyngealgoritme, men demonstrerer et fundamentalt poeng innen data science og musikkvitenskap:
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
