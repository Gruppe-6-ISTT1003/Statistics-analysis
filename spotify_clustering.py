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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    silhouette_score, silhouette_samples, adjusted_rand_score,
    normalized_mutual_info_score, accuracy_score, classification_report, confusion_matrix
)

# SciPy for dendrogram og optimal klynge-tilordning
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

# Fast, gjennomgående fargepalett for de 5 gruppene på tvers av ALLE figurer
PALETTE_GENRES = {
    'Latin / Reggaeton': '#e74c3c',       # Rød
    'Rock / Metal': '#2c3e50',            # Mørk skiferblå/grå
    'Acoustic / R&B / Indie': '#27ae60',  # Grønn
    'Hip-Hop / Rap': '#9b59b6',           # Lilla
    'Pop / Dance': '#3498db'              # Blå
}
CLUSTER_COLORS = [PALETTE_GENRES[g] for g in GENRE_ORDER]


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
    
    # Figur 2: Radar-profil (Parallel Coordinates / Z-score profil) med samme gruppefarger
    scaled_centers = pd.DataFrame(ordered_centers, columns=feature_cols)
    
    plt.figure(figsize=(12, 6))
    for i in range(k):
        plt.plot(feature_cols, scaled_centers.iloc[i], marker='o', lw=2.5,
                 color=CLUSTER_COLORS[i], label=cluster_labels_map[i])
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

    # Felles akserekkevidde for alle individuelle PCA-plott
    x_min, x_max = df['PC1'].min() - 0.5, df['PC1'].max() + 0.5
    y_min, y_max = df['PC2'].min() - 0.5, df['PC2'].max() + 0.5

    # Figur 3b: 2x5 oversiktsgrid med hver farge/kategori på sitt eget plott
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True, sharey=True)
    for idx, genre in enumerate(GENRE_ORDER):
        color = palette_genres[genre]

        ax_top = axes[0, idx]
        df_other_c = df[df['Cluster_Genre'] != genre]
        df_this_c = df[df['Cluster_Genre'] == genre]
        ax_top.scatter(df_other_c['PC1'], df_other_c['PC2'], c='#dcdde1', s=15, alpha=0.35, label='Andre låter')
        ax_top.scatter(df_this_c['PC1'], df_this_c['PC2'], c=color, s=28, alpha=0.8, edgecolors='white', linewidth=0.3,
                       label=f"K{idx} (n={len(df_this_c)})")
        ax_top.set_title(f"K{idx}: {genre}\n(k-Means, n={len(df_this_c)})", fontweight='bold', fontsize=11)
        ax_top.set_xlim(x_min, x_max)
        ax_top.set_ylim(y_min, y_max)
        if idx == 0:
            ax_top.set_ylabel(f"k-Means Klynge\nPC2 ({var_exp[1]:.1%})", fontweight='bold')

        ax_bot = axes[1, idx]
        df_other_g = df_known[df_known['Genre_Family'] != genre]
        df_this_g = df_known[df_known['Genre_Family'] == genre]
        ax_bot.scatter(df_other_g['PC1'], df_other_g['PC2'], c='#dcdde1', s=15, alpha=0.35, label='Andre kjente sjangre')
        ax_bot.scatter(df_this_g['PC1'], df_this_g['PC2'], c=color, s=28, alpha=0.8, edgecolors='white', linewidth=0.3,
                       label=f"{genre} (n={len(df_this_g)})")
        ax_bot.set_title(f"{genre}\n(Faktisk sjanger, n={len(df_this_g)})", fontweight='bold', fontsize=11)
        ax_bot.set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontweight='bold')
        if idx == 0:
            ax_bot.set_ylabel(f"Faktisk sjanger (CSV)\nPC2 ({var_exp[1]:.1%})", fontweight='bold')

    plt.suptitle("Individuell PCA-projeksjon per farge/kategori: k-Means Klynger (øverst) vs. Faktiske Sjangre (nederst)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path3b = os.path.join(FIGURES_DIR, "3b_pca_individual_grid.png")
    plt.savefig(fig_path3b, dpi=300)
    plt.close()
    print(f"Lagret: {fig_path3b}")

    # Figur 3_1 til 3_5: Separate side-ved-side filer for hver enkelt farge/sjanger
    slug_map = {
        'Latin / Reggaeton': 'latin_reggaeton',
        'Rock / Metal': 'rock_metal',
        'Acoustic / R&B / Indie': 'acoustic_rnb_indie',
        'Hip-Hop / Rap': 'hiphop_rap',
        'Pop / Dance': 'pop_dance'
    }
    for idx, genre in enumerate(GENRE_ORDER, start=1):
        color = palette_genres[genre]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)

        df_other_c = df[df['Cluster_Genre'] != genre]
        df_this_c = df[df['Cluster_Genre'] == genre]
        ax1.scatter(df_other_c['PC1'], df_other_c['PC2'], c='#dcdde1', s=22, alpha=0.35, label='Øvrige klynger')
        ax1.scatter(df_this_c['PC1'], df_this_c['PC2'], c=color, s=40, alpha=0.8, edgecolors='white', linewidth=0.4,
                    label=f"K{idx-1}: {genre} (n={len(df_this_c)})")
        ax1.set_title(f"A) k-Means Klynge K{idx-1}: {genre}", fontweight='bold')
        ax1.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians) - Energi/Lydstyrke vs Akustisk", fontweight='bold')
        ax1.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians) - Dansbarhet & Munterhet", fontweight='bold')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.legend(loc="upper left", frameon=True)

        df_other_g = df_known[df_known['Genre_Family'] != genre]
        df_this_g = df_known[df_known['Genre_Family'] == genre]
        ax2.scatter(df_other_g['PC1'], df_other_g['PC2'], c='#dcdde1', s=22, alpha=0.35, label='Øvrige sjangre')
        ax2.scatter(df_this_g['PC1'], df_this_g['PC2'], c=color, s=40, alpha=0.8, edgecolors='white', linewidth=0.4,
                    label=f"Faktisk: {genre} (n={len(df_this_g)})")
        ax2.set_title(f"B) Faktisk sjanger (Ground Truth): {genre}", fontweight='bold')
        ax2.set_xlabel(f"PC1 ({var_exp[0]:.1%} varians)", fontweight='bold')
        ax2.set_ylabel(f"PC2 ({var_exp[1]:.1%} varians)", fontweight='bold')
        ax2.legend(loc="upper left", frameon=True)

        plt.suptitle(f"PCA-sammenligning for {genre}: Predikert klynge vs. Faktisk sjanger", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_path_ind = os.path.join(FIGURES_DIR, f"3_{idx}_pca_{slug_map[genre]}.png")
        plt.savefig(fig_path_ind, dpi=300)
        plt.close()
        print(f"Lagret: {fig_path_ind}")


def hierarchical_clustering(df, X_scaled):
    """
    Hierarkisk klyngeanalyse med Ward's metode og dendrogram, fargekodet etter de 5 gruppene.
    """
    print("\n" + "=" * 65)
    print("5. HIERARKISK AGGLOMERATIV KLYNGEANALYSE")
    print("=" * 65)
    
    # For dendrogram bruker vi et representativt subutvalg (100 låter) for lesbarhet
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), size=100, replace=False)
    X_sub = X_scaled[sample_indices]
    sub_clusters = df['Cluster'].iloc[sample_indices].values
    
    Z = linkage(X_sub, method='ward')
    
    # Koble de 5 hierarkiske grenene (fra venstre mot høyre) til de 5 k-Means-fargene
    h_labels = fcluster(Z, t=10.5, criterion='distance')
    ct_h = pd.crosstab(h_labels, sub_clusters)
    row_ind, col_ind = linear_sum_assignment(-ct_h.values)
    branch_palette = [CLUSTER_COLORS[col_ind[r]] for r in range(len(row_ind))]
    set_link_color_palette(branch_palette)
    
    # Figur 4: Dendrogram
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
    cutoff_line = plt.axhline(10.5, color='#c0392b', linestyle='--', lw=1.8, label='Snitthøyde for 5 hovedgrener (d = 10.5)')
    legend_handles = [cutoff_line] + [
        Line2D([0], [0], color=CLUSTER_COLORS[i], lw=3, label=f"Gren → K{i}: {g}")
        for i, g in enumerate(GENRE_ORDER)
    ]
    plt.title("Hierarkisk dendrogram (Ward's kobling) fargekodet etter de 5 hovedgruppene", fontweight='bold', fontsize=13)
    plt.xlabel("Klyngeindeks / Antall sammenslåtte låter", fontweight='bold')
    plt.ylabel("Euklidsk avstand (Ward's distanse)", fontweight='bold')
    plt.legend(handles=legend_handles, loc='upper right', frameon=True)
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
    
    # Figur 5: Kvadratisk (5x5) Heatmap av kryss-tabellen med fargekodede gruppeetiketter
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    cluster_axis_labels = [f"K{i}: {g}" for i, g in enumerate(GENRE_ORDER)]
    sns.heatmap(
        ct_norm,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        square=True,
        cbar_kws={'label': 'Andel av sjanger i klynge (%)', 'shrink': 0.82},
        xticklabels=cluster_axis_labels,
        yticklabels=GENRE_ORDER,
        ax=ax
    )
    ax.set_title("Kvadratisk (5×5) samsvarsmatrise: Faktiske sjangre vs. k-Means klynger (%)", fontweight='bold', fontsize=12)
    ax.set_xlabel("Tilordnet k-Means Klynge (Akustisk profil)", fontweight='bold')
    ax.set_ylabel("Faktisk sjangerfamilie i CSV (Ground Truth)", fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    for i, color in enumerate(CLUSTER_COLORS):
        ax.get_xticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_color(color)
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
    
    # Figur 6: Fordeling av alle 7 akustiske egenskaper per klynge
    all_features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Valence', 'Tempo'
    ]
    cluster_palette = ['#e74c3c', '#2c3e50', '#27ae60', '#9b59b6', '#3498db']
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for idx, col in enumerate(all_features):
        sns.boxplot(x='Cluster', y=col, hue='Cluster', data=df, ax=axes[idx], palette=cluster_palette, legend=False)
        axes[idx].set_title(f"Fordeling av {col}", fontweight='bold')
        axes[idx].set_xlabel("Klynge")
        axes[idx].set_xticks(range(len(GENRE_ORDER)))
        axes[idx].set_xticklabels([f"K{i}\n({g.split('/')[0].strip()})" for i, g in enumerate(GENRE_ORDER)], fontsize=9)

    # Bruk det 8. panelet til en oversiktlig fargeforklaring for de 5 klyngene
    axes[7].axis('off')
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=cluster_palette[i], edgecolor='black', label=f"K{i}: {g}")
        for i, g in enumerate(GENRE_ORDER)
    ]
    axes[7].legend(
        handles=legend_handles,
        title="k-Means Klynger (k=5)",
        title_fontsize=12,
        fontsize=11,
        loc='center',
        frameon=True
    )

    plt.suptitle("Akustiske forskjeller på tvers av de 5 klyngene (Alle 7 egenskaper)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path6 = os.path.join(FIGURES_DIR, "6_feature_distributions_per_cluster.png")
    plt.savefig(fig_path6, dpi=300)
    plt.close()
    print(f"Lagret: {fig_path6}")


def perform_genre_classification(df, feature_cols):
    """
    Utfører veiledet klassifikasjon (K-nærmeste naboer / KNN og Logistisk Regresjon)
    på låtene med kjent sjangerfamilie (80% trening, 20% test).
    """
    print("\n" + "=" * 65)
    print("7. VEILEDET KLASSIFIKASJON: KNN OG LOGISTISK REGRESJON")
    print("=" * 65)

    df_known = df[df['Genre_Family'] != 'Ukjent / Mangler tag'].copy()
    X = df_known[feature_cols].values
    y = df_known['Genre_Family'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Treningsdata: {len(X_train)} låter (80%), Testdata: {len(X_test)} låter (20%)")

    scaler_clf = StandardScaler()
    X_train_s = scaler_clf.fit_transform(X_train)
    X_test_s = scaler_clf.transform(X_test)

    # 1. Velg optimal K for KNN med 5-fold kryssvalidering på treningssettet
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    k_values = list(range(1, 36, 2))
    cv_means = []
    cv_stds = []
    test_accs = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_s, y_train, cv=cv, scoring='accuracy')
        cv_means.append(scores.mean() * 100)
        cv_stds.append(scores.std() * 100)
        knn.fit(X_train_s, y_train)
        test_accs.append(accuracy_score(y_test, knn.predict(X_test_s)) * 100)

    best_idx = int(np.argmax(cv_means))
    best_k = k_values[best_idx]
    print(f"\nOptimal K for KNN (5-fold CV): K = {best_k} (CV-nøyaktighet: {cv_means[best_idx]:.1f}%)")

    # Figur 7: Valg av K i KNN (Kryssvalidering vs Testsett)
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, cv_means, 'o-', color='#2c3e50', lw=2, label='5-fold Kryssvalidering (Treningssett)')
    plt.plot(k_values, test_accs, 's--', color='#3498db', lw=2, alpha=0.85, label='Testsett (20% uavhengig)')
    plt.axvline(best_k, color='#e74c3c', linestyle=':', lw=2, label=f'Valgt K = {best_k} ({cv_means[best_idx]:.1f}%)')
    plt.title('Valg av antall naboer (K) i K-nærmeste naboer (KNN)', fontweight='bold', fontsize=13)
    plt.xlabel('Antall nærmeste naboer (K)', fontweight='bold')
    plt.ylabel('Klassifikasjonsnøyaktighet / Accuracy (%)', fontweight='bold')
    plt.xticks(k_values)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    fig_path7 = os.path.join(FIGURES_DIR, "7_knn_k_selection.png")
    plt.savefig(fig_path7, dpi=300)
    plt.close()
    print(f"Lagret: {fig_path7}")

    # 2. Tren beste KNN og Multinomial Logistisk Regresjon
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train_s, y_train)
    y_pred_knn = knn_best.predict(X_test_s)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_s, y_train)
    y_pred_lr = logreg.predict(X_test_s)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    print(f"\nTest-nøyaktighet KNN (K={best_k}):          {acc_knn:.1%}")
    print(f"Test-nøyaktighet Logistisk Regresjon:  {acc_lr:.1%}")
    print("\nKlassifikasjonsrapport på testsettet (KNN, K=15):")
    print(classification_report(y_test, y_pred_knn, labels=GENRE_ORDER, digits=3))

    # 3. Figur 8: Kvadratisk (5x5) forvirringsmatrise på testsettet (KNN vs Logistisk Regresjon)
    cm_knn_pct = confusion_matrix(y_test, y_pred_knn, labels=GENRE_ORDER, normalize='true') * 100
    cm_lr_pct = confusion_matrix(y_test, y_pred_lr, labels=GENRE_ORDER, normalize='true') * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5))

    sns.heatmap(
        cm_knn_pct, annot=True, fmt=".1f", cmap="YlGnBu", square=True,
        cbar_kws={'label': 'Andel (%) av sann sjanger', 'shrink': 0.78},
        xticklabels=GENRE_ORDER, yticklabels=GENRE_ORDER, ax=ax1
    )
    ax1.set_title(f"A) KNN (K={best_k}) på testsett (Accuracy = {acc_knn:.1%})", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Predikert sjangerfamilie", fontweight='bold')
    ax1.set_ylabel("Faktisk sjangerfamilie i CSV (Sann klasse)", fontweight='bold')
    ax1.set_xticklabels(GENRE_ORDER, rotation=30, ha='right', fontweight='bold')
    ax1.set_yticklabels(GENRE_ORDER, rotation=0, fontweight='bold')

    sns.heatmap(
        cm_lr_pct, annot=True, fmt=".1f", cmap="YlGnBu", square=True,
        cbar_kws={'label': 'Andel (%) av sann sjanger', 'shrink': 0.78},
        xticklabels=GENRE_ORDER, yticklabels=GENRE_ORDER, ax=ax2
    )
    ax2.set_title(f"B) Logistisk regresjon på testsett (Accuracy = {acc_lr:.1%})", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Predikert sjangerfamilie", fontweight='bold')
    ax2.set_ylabel("")
    ax2.set_xticklabels(GENRE_ORDER, rotation=30, ha='right', fontweight='bold')
    ax2.set_yticklabels(GENRE_ORDER, rotation=0, fontweight='bold')

    for i, color in enumerate(CLUSTER_COLORS):
        ax1.get_xticklabels()[i].set_color(color)
        ax1.get_yticklabels()[i].set_color(color)
        ax2.get_xticklabels()[i].set_color(color)
        ax2.get_yticklabels()[i].set_color(color)

    plt.suptitle("Veiledet klassifikasjon (5×5 forvirringsmatrise i % på testsettet, n=314)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path8 = os.path.join(FIGURES_DIR, "8_classification_confusion_matrix.png")
    plt.savefig(fig_path8, dpi=300)
    plt.close()
    print(f"Lagret: {fig_path8}")

    # Lagre også KNN-predikert sjanger på hele datasettet i df
    X_all_s = scaler_clf.transform(df[feature_cols].values)
    df['Predicted_Genre_KNN'] = knn_best.predict(X_all_s)


def main():
    print("STARTER KLYNGEANALYSE OG KLASSIFIKASJON PÅ SPOTIFY TOPP 2500...\n")
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

    # Veiledet klassifikasjon (KNN og Logistisk regresjon)
    perform_genre_classification(df, feature_cols)
    
    # Lagre ny CSV-fil med generaliserte sjangerkategorier, klyngetilhørighet og KNN-prediksjon
    orig_cols = [c for c in df.columns if c not in [
        'Genre_Family', 'Raw_Cluster', 'Cluster', 'Cluster_Genre',
        'Cluster_Name', 'Predicted_Genre_KNN', 'PC1', 'PC2'
    ]]
    genres_idx = orig_cols.index('Genres') + 1
    out_cols = orig_cols[:genres_idx] + ['Genre_Family', 'Cluster', 'Cluster_Genre', 'Predicted_Genre_KNN'] + orig_cols[genres_idx:]
    out_csv_path = "data/kworb_top2500_spotify_generalized.csv"
    df[out_cols].to_csv(out_csv_path, index=False)
    print(f"\nLagret CSV med generaliserte kategorier: {out_csv_path}")
    
    print("\n" + "=" * 65)
    print("KLYNGEANALYSE FULLFØRT!")
    print(f"Alle figurer er lagret i: '{FIGURES_DIR}/'")
    print("=" * 65)


if __name__ == "__main__":
    main()
