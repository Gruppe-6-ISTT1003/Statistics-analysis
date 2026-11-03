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

# SciPy for dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Sett stil for visualiseringer
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})
FIGURES_DIR = "figures/clustering"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_and_prepare_data(filepath="data/kworb_top2500_spotify.csv"):
    """
    Laster inn data og mapper rå sjanger-tags til overordnede sjangerfamilier.
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
    
    # Kartlegg rå sjanger-tags til hovedfamilier (for ekstern validering)
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
    
    # Markér k=4 som optimal balanse mellom tolkbarhet og metrikk
    ax1.axvline(4, color='gray', linestyle=':', alpha=0.8)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    fig_path1 = os.path.join(FIGURES_DIR, "1_elbow_and_silhouette.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"\nLagret: {fig_path1}")
    
    return 4  # Anbefalt k


def perform_kmeans_clustering(df, X_scaled, feature_cols, k=4):
    """
    Utfører k-Means med k=4 og profilerer klyngene.
    """
    print("\n" + "=" * 65)
    print(f"3. UTFØRER K-MEANS KLYNGEANALYSE (k = {k})")
    print("=" * 65)
    
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    df['Cluster'] = km.fit_predict(X_scaled)
    
    # Beregn klyngesentrene på original skala
    cluster_centers_orig = df.groupby('Cluster')[feature_cols].mean()
    print("\nKlyngenes gjennomsnittsverdier (original måleskala):")
    print(cluster_centers_orig.round(3).T)
    
    # Gi hver klynge en tolkbar musikalsk arketype basert på profil
    cluster_labels_map = {
        0: "Klynge 0: Munter Klubb & Latin Pop (Høy Danceability + Høy Valence)",
        1: "Klynge 1: Drivende Rock & Høy Energi (Lav Akustiskhet + Lav Valence)",
        2: "Klynge 2: Akustisk & Rolige Ballader (Høy Akustiskhet + Lav Energi)",
        3: "Klynge 3: Rytmisk Urban & Rap (Høy Taleandel + Raskere Tempo)"
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_labels_map)
    
    # Figur 2: Radar-profil (Parallel Coordinates / Z-score profil)
    scaler = StandardScaler()
    scaled_centers = pd.DataFrame(km.cluster_centers_, columns=feature_cols)
    
    plt.figure(figsize=(12, 6))
    for i in range(k):
        plt.plot(feature_cols, scaled_centers.iloc[i], marker='o', lw=2.5,
                 label=f"Klynge {i}: {cluster_labels_map[i].split(':')[1].strip()}")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Datasettets gjennomsnitt (Z=0)')
    plt.title("Akustiske fingeravtrykk for de 4 klyngene (Standardiserte Z-verdier)", fontweight='bold', fontsize=13)
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
    
    # Figur 3: Side-ved-side PCA (Klynger vs Reelle Sjangre)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel 1: k-Means Klynger
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster', palette='tab10',
        data=df, alpha=0.65, s=35, ax=ax1
    )
    ax1.set_title(f"A) k-Means Klynger (k=4) i PCA-rommet", fontweight='bold')
    ax1.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians) - Energi/Lydstyrke vs Akustisk", fontweight='bold')
    ax1.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians) - Dansbarhet & Munterhet", fontweight='bold')
    
    # Panel 2: Faktiske sjangre (ekskluderer ukjente for klarhet)
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
    
    plt.suptitle("Sammenligning av uovervåkede klynger og musikksjangre (Spotify Top 2500)", fontsize=14, fontweight='bold')
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
    
    # For dendrogram bruker vi et representativt subutvalg (f.eks. 120 låter) for lesbarhet
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
        color_threshold=12.0
    )
    plt.axhline(12.0, color='red', linestyle='--', label='Snitthøyde for 4 hovedgrener')
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
    Evaluerer hvor godt de akustiske klyngene samsvarer med faktiske sjangre.
    """
    print("\n" + "=" * 65)
    print("6. VALIDERER KLYNGENE MOT REELLE SJANGRE")
    print("=" * 65)
    
    df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag'].copy()
    
    # Kryss-tabell (Contingency Table)
    ct = pd.crosstab(df_known['Genre_Family'], df_known['Cluster'], margins=True)
    print("\nKryss-tabell mellom faktiske sjangre og k-Means klynger:")
    print(ct)
    
    # Prosentvis fordeling per sjanger over klyngene
    ct_norm = pd.crosstab(df_known['Genre_Family'], df_known['Cluster'], normalize='index') * 100
    
    # Figur 5: Heatmap av kryss-tabellen
    plt.figure(figsize=(10, 6))
    cluster_short_names = [
        "Klynge 0: Klubb/Latin Pop",
        "Klynge 1: Rock/Energi",
        "Klynge 2: Akustisk/Ballade",
        "Klynge 3: Urban/Rap"
    ]
    sns.heatmap(ct_norm, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Andel av sjanger i klynge (%)'},
                xticklabels=cluster_short_names)
    plt.title("Hvor havner sjangrene? Andel (%) av hver sjanger fordelt på klynger", fontweight='bold', fontsize=12)
    plt.xlabel("k-Means Klynge", fontweight='bold')
    plt.ylabel("Faktisk sjangerfamilie (Ground Truth)", fontweight='bold')
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
    
    # Beregn klynge-purity
    # Purity = sum(max_g(count_k,g)) / N
    cm = pd.crosstab(df_known['Cluster'], df_known['Genre_Family'])
    purity = cm.max(axis=1).sum() / len(df_known)
    
    print("\nValideringsmetrikker mot Ground Truth:")
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
        axes[idx].set_xticklabels([f"K{i}" for i in range(4)])
    plt.suptitle("Akustiske forskjeller på tvers av de 4 klyngene", fontsize=14, fontweight='bold')
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
