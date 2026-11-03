"""
ISTT1003 Statistikk, statistisk læring og data science
Prosjekt: Klyngeanalyse (Cluster Analysis) for å identifisere musikksjangre fra Spotify-lydegenskaper

Metoder:
1. Dataklargjøring & sjangerkategorisering (ground truth)
2. Z-score standardisering (StandardScaler)
3. k-Means klyngeanalyse med optimalisering av k (Elbow-metoden & Silhuettanalyse)
4. Hierarkisk klyngeanalyse (Agglomerative Clustering med Ward's kobling & dendrogram)
5. Dimensjonsreduksjon med PCA (Principal Component Analysis) for 2D visualisering
6. Ekstern validering mot reelle sjangerfamilier (Kryss-tabell, Purity, ARI, NMI)
7. Karakterisering av musikkprofiler (akustiske fingeravtrykk)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score

# SciPy for dendrogram og optimal klynge-tilordning
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment

# Sett stil for visualiseringer
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})
FIGURES_DIR = "figures/clustering"
os.makedirs(FIGURES_DIR, exist_ok=True)

GENRE_ORDER = [
    'Latin / Reggaeton',
    'Rock / Metal',
    'Acoustic / R&B / Indie',
    'Hip-Hop / Rap',
    'Pop / Dance'
]


def load_and_prepare_data(filepath="data/kworb_top2500_spotify.csv"):
    """
    Laster inn data og mapper rå sjanger-tags til 5 overordnede sjangerfamilier.
    """
    print("=" * 65)
    print("1. LASTER INN OG FORBEREDER DATA FOR KLYNGEANALYSE")
    print("=" * 65)
    df = pd.read_csv(filepath)
    print(f"Totalt antall låter: {len(df)}")
    
    # Utvalgte akustiske egenskaper (kontinuerlige målinger)
    feature_cols = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Valence', 'Tempo'
    ]
    
    # Kartlegg rå sjanger-tags til 5 hovedfamilier (for 1-til-1 validering mot k=5 klynger)
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
    
    print("\nFordeling av sjangerfamilier (Ground Truth):")
    genre_dist = df['Genre_Family'].value_counts()
    for g, count in genre_dist.items():
        print(f"  {g:<24}: {count:>4} låter ({count/len(df):.1%})")
        
    return df, feature_cols


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """
    Kjører Elbow-metoden (Inertia/WCSS) og beregner Silhuett-koeffisienter for k fra 2 til 10.
    """
    print("\n" + "=" * 65)
    print("2. FINNER OPTIMALT ANTALL KLYNGER (k) FOR K-MEANS")
    print("=" * 65)
    
    inertias = []
    sil_scores = []
    
    print(f"{'k':<4} | {'Inertia (WCSS)':<16} | {'Silhuett-score':<15}")
    print("-" * 40)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        inertias.append(inertia)
        sil_scores.append(sil)
        print(f"{k:<4} | {inertia:<16.1f} | {sil:<15.3f}")
        
    # Figur 1: Elbow- og Silhuett-kurver
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
    
    plt.title('Valg av optimalt k: Elbow-metoden og Silhuett-analyse', fontweight='bold', fontsize=13)
    
    # Markér k=5 for direkte sammenligning med de 5 sjangerfamiliene
    ax1.axvline(5, color='gray', linestyle=':', alpha=0.8, label='Valgt k=5 (5 sjangerfamilier)')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    fig_path1 = os.path.join(FIGURES_DIR, "1_elbow_and_silhouette.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"\nLagret: {fig_path1}")
    
    return 5  # k=5 matcher de 5 sjangerfamiliene


def perform_kmeans_clustering(df, X_scaled, feature_cols, k=5):
    """
    Utfører k-Means med k=5, tilordner klyngene 1-til-1 mot de 5 sjangerfamiliene,
    og profilerer klyngene.
    """
    print("\n" + "=" * 65)
    print(f"3. UTFØRER K-MEANS KLYNGEANALYSE (k = {k})")
    print("=" * 65)
    
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    raw_clusters = km.fit_predict(X_scaled)
    df['Raw_Cluster'] = raw_clusters
    
    # Tilordne hver klynge 1-til-1 til sjangerfamilien den samsvarer best med (Hungarian matching)
    df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag']
    ct_raw = pd.crosstab(df_known['Genre_Family'], df_known['Raw_Cluster'], normalize='index').reindex(GENRE_ORDER)
    row_ind, col_ind = linear_sum_assignment(-ct_raw.values)
    remap = {int(col_ind[r]): r for r in row_ind}
    
    df['Cluster'] = df['Raw_Cluster'].map(remap)
    df['Cluster_Genre'] = df['Cluster'].map(lambda c: GENRE_ORDER[c])
    
    # Reorganiser klyngesentrene i samme rekkefølge som GENRE_ORDER (0..4)
    inv_remap = {new_idx: old_idx for old_idx, new_idx in remap.items()}
    ordered_centers = np.array([km.cluster_centers_[inv_remap[i]] for i in range(k)])
    
    # Beregn klyngesentrene på original skala
    cluster_centers_orig = df.groupby('Cluster')[feature_cols].mean()
    print("\nKlyngenes gjennomsnittsverdier (original måleskala):")
    print(cluster_centers_orig.round(3).T)
    
    # Koble hver klynge til sin tilsvarende sjangerkategori og akustiske profil
    cluster_labels_map = {
        0: "K0: Latin / Reggaeton (Høy Danceability + Høy Valence)",
        1: "K1: Rock / Metal (Høy Energi + Raskt Tempo)",
        2: "K2: Acoustic / R&B / Indie (Høy Akustiskhet + Lav Energi)",
        3: "K3: Hip-Hop / Rap (Høy Speechiness + Rytmisk)",
        4: "K4: Pop / Dance (Balansert Energi & Dansbarhet)"
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_labels_map)
    
    # Figur 2: Radar-profil (Parallel Coordinates / Z-score profil)
    scaled_centers = pd.DataFrame(ordered_centers, columns=feature_cols)
    
    plt.figure(figsize=(12, 6))
    for i in range(k):
        plt.plot(feature_cols, scaled_centers.iloc[i], marker='o', lw=2.5,
                 label=cluster_labels_map[i])
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Datasettets gjennomsnitt (Z=0)')
    plt.title("Akustiske fingeravtrykk for de 5 klyngene (Standardiserte Z-verdier)", fontweight='bold', fontsize=13)
    plt.xlabel("Akustisk egenskap", fontweight='bold')
    plt.ylabel("Standardavvik fra gjennomsnitt (Z-score)", fontweight='bold')
    plt.xticks(rotation=15)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig_path2 = os.path.join(FIGURES_DIR, "2_cluster_radar_profiles.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Lagret: {fig_path2}")
    
    return km


def perform_pca_and_visualization(df, X_scaled, feature_cols):
    """
    Dimensjonsreduksjon med PCA og side-ved-side projeksjon av Klynger vs. Reelle Sjangre.
    """
    print("\n" + "=" * 65)
    print("4. DIMENSJONSREDUKSJON MED PCA FOR 2D-VISUALISERING")
    print("=" * 65)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    
    var_exp = pca.explained_variance_ratio_
    print(f"Forklart varians: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%} (Totalt {sum(var_exp):.1%})")
    
    pca_loadings = pd.DataFrame(pca.components_, columns=feature_cols, index=['PC1', 'PC2'])
    print("\nPCA Hovedkomponenter (Loadings):")
    print(pca_loadings.round(3))
    
    # Felles fargepalett for de 5 kategoriene i både Panel A og Panel B
    palette_genres = {
        'Latin / Reggaeton': '#e74c3c',
        'Rock / Metal': '#2c3e50',
        'Acoustic / R&B / Indie': '#27ae60',
        'Hip-Hop / Rap': '#9b59b6',
        'Pop / Dance': '#3498db'
    }
    
    # Figur 3: Side-ved-side PCA (Klynger vs Reelle Sjangre)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: k-Means Klynger (mappet til samme 5 kategorinavn og farger)
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster_Genre', hue_order=GENRE_ORDER, palette=palette_genres,
        data=df, alpha=0.65, s=35, ax=ax1
    )
    ax1.set_title("A) k-Means Klynger (k=5) i PCA-rommet", fontweight='bold')
    ax1.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians) - Energi/Lydstyrke vs Akustisk", fontweight='bold')
    ax1.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians) - Dansbarhet & Munterhet", fontweight='bold')
    ax1.legend(title="Predikert klynge", loc="best")
    
    # Panel 2: Faktiske sjangre (ekskluderer ukjente for klarhet)
    df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag'].copy()
    sns.scatterplot(
        x='PC1', y='PC2', hue='Genre_Family', hue_order=GENRE_ORDER, palette=palette_genres,
        data=df_known, alpha=0.65, s=35, ax=ax2
    )
    ax2.set_title("B) Faktiske sjangerfamilier (Ground Truth) i PCA-rommet", fontweight='bold')
    ax2.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians)", fontweight='bold')
    ax2.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians)", fontweight='bold')
    ax2.legend(title="Faktisk sjanger", loc="best")
    
    plt.suptitle("Sammenligning av uovervåkede klynger (k=5) og musikksjangre (Spotify Top 2500)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path3 = os.path.join(FIGURES_DIR, "3_pca_clusters_vs_genres.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Lagret: {fig_path3}")


def hierarchical_clustering(df, X_scaled):
    """
    Hierarkisk klyngeanalyse med Ward's metode og dendrogram.
    """
    print("\n" + "=" * 65)
    print("5. HIERARKISK AGGLOMERATIV KLYNGEANALYSE")
    print("=" * 65)
    
    # For dendrogram bruker vi et representativt subutvalg (100 låter) for lesbarhet
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=100, replace=False)
    X_sub = X_scaled[sample_indices]
    
    Z = linkage(X_sub, method='ward')
    
    # Figur 4: Dendrogram
    plt.figure(figsize=(14, 7))
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
        color_threshold=10.5
    )
    plt.axhline(10.5, color='red', linestyle='--', label='Snitthøyde for 5 hovedgrener')
    plt.title("Hierarkisk dendrogram (Ward's kobling) på et tilfeldig utvalg låter", fontweight='bold', fontsize=13)
    plt.xlabel("Klyngeindeks / Antall sammenslåtte låter", fontweight='bold')
    plt.ylabel("Euklidsk avstand (Ward's distanse)", fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    fig_path4 = os.path.join(FIGURES_DIR, "4_hierarchical_dendrogram.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"Lagret: {fig_path4}")


def validate_against_genres(df):
    """
    Evaluerer hvor godt de 5 akustiske klyngene samsvarer med de 5 faktiske sjangerfamiliene
    i en kvadratisk (5x5) kontingenstabell / forvirringsmatrise.
    """
    print("\n" + "=" * 65)
    print("6. VALIDERER KLYNGENE MOT REELLE SJANGRE (5x5 MATRISE)")
    print("=" * 65)
    
    df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag'].copy()
    
    # Kryss-tabell (Contingency Table) med identisk rekkefølge på rader og kolonner (5x5)
    ct = pd.crosstab(df_known['Genre_Family'], df_known['Cluster_Genre']).reindex(
        index=GENRE_ORDER, columns=GENRE_ORDER
    )
    print("\nKvadratisk kryss-tabell (5x5) mellom faktiske sjangre og k-Means klynger:")
    print(ct)
    
    # Prosentvis fordeling per sjanger over klyngene (5x5)
    ct_norm = pd.crosstab(
        df_known['Genre_Family'], df_known['Cluster_Genre'], normalize='index'
    ).reindex(index=GENRE_ORDER, columns=GENRE_ORDER) * 100
    
    # Figur 5: Kvadratisk (5x5) Heatmap av kryss-tabellen
    plt.figure(figsize=(9, 8))
    cluster_axis_labels = [f"K{i}: {g}" for i, g in enumerate(GENRE_ORDER)]
    sns.heatmap(
        ct_norm,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        square=True,
        cbar_kws={'label': 'Andel av sjanger i klynge (%)', 'shrink': 0.82},
        xticklabels=cluster_axis_labels,
        yticklabels=GENRE_ORDER
    )
    plt.title("Kvadratisk (5×5) samsvarsmatrise: Faktiske sjangre vs. k-Means klynger (%)", fontweight='bold', fontsize=12)
    plt.xlabel("Tilordnet k-Means Klynge (Akustisk profil)", fontweight='bold')
    plt.ylabel("Faktisk sjangerfamilie i CSV (Ground Truth)", fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig_path5 = os.path.join(FIGURES_DIR, "5_cluster_genre_heatmap.png")
    plt.savefig(fig_path5, dpi=300)
    plt.close()
    print(f"\nLagret: {fig_path5}")
    
    # Beregn valideringsmetrikker
    labels_true = df_known['Genre_Family'].astype('category').cat.codes
    labels_pred = df_known['Cluster']
    
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    
    # Beregn klynge-purity og diagonal treffsikkerhet
    cm = pd.crosstab(df_known['Cluster'], df_known['Genre_Family'])
    purity = cm.max(axis=1).sum() / len(df_known)
    diag_acc = np.trace(ct.values) / len(df_known)
    
    print("\nValideringsmetrikker mot Ground Truth:")
    print(f"  Diagonal 1-til-1 treff (5x5):   {diag_acc:.1%}")
    print(f"  Cluster Purity:                 {purity:.1%}")
    print(f"  Adjusted Rand Index (ARI):      {ari:.3f}")
    print(f"  Normalized Mutual Info (NMI):   {nmi:.3f}")
    
    # Figur 6: Fordeling av nøkkelegenskaper per klynge
    key_features = ['Danceability', 'Speechiness', 'Acousticness', 'Energy']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, col in enumerate(key_features):
        sns.boxplot(x='Cluster', y=col, hue='Cluster', data=df, ax=axes[idx], palette='tab10', legend=False)
        axes[idx].set_title(f"Fordeling av {col}", fontweight='bold')
        axes[idx].set_xlabel("Klynge")
        axes[idx].set_xticks(range(len(GENRE_ORDER)))
        axes[idx].set_xticklabels([f"K{i}\n({g.split('/')[0].strip()})" for i, g in enumerate(GENRE_ORDER)], fontsize=9)
    plt.suptitle("Akustiske forskjeller på tvers av de 5 klyngene", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path6 = os.path.join(FIGURES_DIR, "6_feature_distributions_per_cluster.png")
    plt.savefig(fig_path6, dpi=300)
    plt.close()
    print(f"Lagret: {fig_path6}")


def main():
    print("STARTER KLYNGEANALYSE PÅ SPOTIFY TOPP 2500 FOR SJANGERIDENTIFIKASJON...\n")
    df, feature_cols = load_and_prepare_data()
    
    # Standardiser egenskapene (Z-score er avgjørende for avstandsbasert clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # Finner optimalt k
    optimal_k = find_optimal_k(X_scaled)
    
    # Utfør k-Means
    km_model = perform_kmeans_clustering(df, X_scaled, feature_cols, k=optimal_k)
    
    # PCA
    perform_pca_and_visualization(df, X_scaled, feature_cols)
    
    # Hierarkisk clustering
    hierarchical_clustering(df, X_scaled)
    
    # Validering mot reelle sjangre
    validate_against_genres(df)
    
    print("\n" + "=" * 65)
    print("KLYNGEANALYSE FULLFØRT!")
    print(f"Alle figurer er lagret i: '{FIGURES_DIR}/'")
    print("=" * 65)


if __name__ == "__main__":
    main()
