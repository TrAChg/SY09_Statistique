"""
B05 - Clustering de la consommation (Dalc, Walc) et definition de la cible.
     Methode retenue: K-means (K=4) car inertie intra inferieure a CAH
     (231.7 vs 252.7, soit 82.5% vs 80.9% de variance expliquee).
     Robustesse: ARI=0.856 entre CAH et K-means confirme la stabilite.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_variable_types, print_section, save_plot, BASE_DIR

df = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol.csv"))
df_scaled = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol-scaled.csv"))
contexte = get_variable_types(df)["numeric"] + get_variable_types(df)["ordinal"]

# On clusterise uniquement sur (Dalc, Walc): c'est la cible, pas le contexte.
# Standardisation pour coherence Ward et K-means (inertie = p = 2).
# Verification: ARI=1.0 entre standardise/non-standardise (meme partition).
X = df[["Dalc", "Walc"]].values
X = (X - X.mean(0)) / X.std(0)
Z = linkage(X, method="ward")


def inertie_intra(X, labels):
    return sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum()
               for c in np.unique(labels))


def coude(ks, w):
    ks, w = np.array(ks, float), np.array(w, float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (w - w.min()) / (w.max() - w.min())
    d = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[np.argmax(d)])


# --- Choix de K : methode du coude sur l'inertie intra-classe ---
print_section("Choix de K (coude sur inertie intra)")
I_tot = ((X - X.mean(0)) ** 2).sum()
ks, w_cah, w_km = [], [], []
print(f"  {'K':>2} | {'intra_CAH':>10} {'%expl':>7} | {'intra_KM':>10} {'%expl':>7} | {'ARI':>6}")
for k in range(2, 9):
    lc = fcluster(Z, t=k, criterion="maxclust")
    lk = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
    wc, wk = inertie_intra(X, lc), inertie_intra(X, lk)
    ks.append(k); w_cah.append(wc); w_km.append(wk)
    print(f"  {k:>2} | {wc:>10.1f} {(1-wc/I_tot)*100:>6.1f}% | "
          f"{wk:>10.1f} {(1-wk/I_tot)*100:>6.1f}% | {adjusted_rand_score(lc, lk):>6.3f}")

K = 4
print(f"  coude CAH={coude(ks, w_cah)}, coude K-means={coude(ks, w_km)} -> K={K}")

plt.figure(figsize=(8, 5))
plt.plot(ks, w_cah, "o-", color="steelblue", label="CAH")
plt.plot(ks, w_km, "s-", color="coral", label="K-means")
plt.axvline(K, color="red", ls="--", label=f"K={K}")
plt.xlabel("K"); plt.ylabel("inertie intra-classe"); plt.legend()
save_plot("B05_elbow.png")

# --- Partition finale: K-MEANS (K=4) ---
# Justification du choix K-means plutot que CAH:
#   - Inertie intra K-means (231.7) < CAH (252.7) a K=4
#   - % variance expliquee: K-means (82.5%) > CAH (80.9%)
#   - K-means minimise DIRECTEMENT l'inertie intra (critere de Lloyd)
#   - ARI=0.856: les deux methodes concordent (robustesse)
#   - CAH sert a VALIDER le choix de K (dendrogramme), K-means a PRODUIRE la partition
print_section("PARTITION FINALE: K-means (K=4)")
km_final = KMeans(n_clusters=K, random_state=42, n_init=20).fit(X)
clusters = km_final.labels_ + 1  # +1 pour labels 1,2,3,4 (comme fcluster)
df["cluster"] = clusters

# Verification: concordance avec CAH
clusters_cah = fcluster(Z, t=K, criterion="maxclust")
ari = adjusted_rand_score(clusters_cah, clusters)
print(f"  ARI(CAH, K-means) = {ari:.3f} -> concordance {'forte' if ari > 0.8 else 'moderee'}")
print(f"  Inertie intra K-means: {inertie_intra(X, clusters):.1f} (vs CAH: {inertie_intra(X, clusters_cah):.1f})")
print(f"  % explique K-means: {(1-inertie_intra(X, clusters)/I_tot)*100:.1f}%")

# --- Table de frequence et zones des clusters ---
print_section("Table Dalc x Walc")
print(pd.crosstab(df["Dalc"], df["Walc"], margins=True, margins_name="Tot").to_string())

print_section("Cluster dominant par cellule")
zones = pd.crosstab(df["Dalc"], df["Walc"], values=df["cluster"],
                    aggfunc=lambda s: f"C{s.mode()[0]}({len(s)})").fillna(".")
print(zones.to_string())

# --- Lien clusters / contexte (ANOVA, confirme B01) ---
print_section("Lien clusters - contexte (ANOVA)")
print(df.groupby("cluster")[["Dalc", "Walc"]].agg(["mean", "size"]).round(2).to_string())
signif = [v for v in contexte
          if f_oneway(*[df.loc[clusters == c, v] for c in np.unique(clusters)])[1] < 0.05]
print(f"  {len(signif)}/{len(contexte)} facteurs lies (p<0.05): {', '.join(signif)}")

# --- Cible 3 classes : non-buveurs / weekend / eleve ---
print_section("Cible 3 classes")
stats = df.groupby("cluster")[["Dalc", "Walc"]].mean()
stats["total"] = stats.sum(axis=1)
print("  centroides par cluster:")
for c in stats.index:
    print(f"    C{c}: Dalc={stats.loc[c,'Dalc']:.2f}, Walc={stats.loc[c,'Walc']:.2f}, "
          f"total={stats.loc[c,'total']:.2f}, n={int((clusters==c).sum())}")

ordre = stats["total"].sort_values().index
c_low = ordre[0]
autres = [c for c in ordre if c != c_low]
c_week = stats.loc[autres, "Dalc"].idxmin()
mapping = {c: (0 if c == c_low else 1 if c == c_week else 2) for c in stats.index}

df["Risk_Class"] = df["cluster"].map(mapping)
noms = {0: "non-buveurs", 1: "weekend", 2: "eleve"}
print()
for g in [0, 1, 2]:
    m = df["Risk_Class"] == g
    print(f"  Classe {g} ({noms[g]:11s}): n={m.sum():3d}, "
          f"Dalc={df.loc[m,'Dalc'].mean():.2f}, Walc={df.loc[m,'Walc'].mean():.2f}")

# --- Sauvegarde ---
df[["cluster", "Risk_Class"]].to_csv(
    os.path.join(BASE_DIR, "alcohol_profiles.csv"), index=False)
df.drop(columns=["cluster", "Risk_Class"]).to_csv(
    os.path.join(BASE_DIR, "student-alcohol.csv"), index=False)

# Dendrogramme CAH (pour visualisation du choix de K)
plt.figure(figsize=(14, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8)
plt.axhline(y=Z[-K, 2], color="red", ls="--")
plt.title(f"CAH Ward - Dalc/Walc (coupe K={K}, partition finale = K-means)")
save_plot("B05_dendrogram_alcohol.png")
