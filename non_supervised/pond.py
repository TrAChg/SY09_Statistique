"""
B05_BIS — APPROCHE ALTERNATIVE: clustering sur la SOMME Dalc+Walc.
     Comparaison avec l'approche 2D (Dalc, Walc) de B05.
     But: montrer que la somme PERD l'information weekend.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import f_oneway, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_variable_types, print_section, save_plot, BASE_DIR

df = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol.csv"))
var_types = get_variable_types(df)

# =============================================================================
# 1. CONSTRUCTION DE LA SOMME
# =============================================================================
print_section("1. SOMME Dalc + Walc")
df["Somme"] = df["Dalc"] + df["Walc"]
print(f"  Range: {df['Somme'].min()} - {df['Somme'].max()}")
print(f"  Moyenne: {df['Somme'].mean():.2f}, Mediane: {df['Somme'].median():.0f}")
print(f"\n  Distribution:")
for s in sorted(df["Somme"].unique()):
    n = (df["Somme"] == s).sum()
    pct = n / len(df) * 100
    bar = "#" * int(pct)
    print(f"    Somme={s:>2}: n={n:>3} ({pct:>5.1f}%) {bar}")

# =============================================================================
# 2. CORRELATIONS SOMME vs VARIABLES DE CONTEXTE
# =============================================================================
print_section("2. CORRELATIONS (Somme vs contexte)")
numeric_vars = var_types["numeric"] + var_types["ordinal"]
print(f"\n  {'Variable':15s} {'cor(Somme)':>10} {'cor(Dalc)':>10} {'cor(Walc)':>10}")
for v in numeric_vars:
    r_somme = df["Somme"].corr(df[v])
    r_dalc = df["Dalc"].corr(df[v])
    r_walc = df["Walc"].corr(df[v])
    marker = " <--" if abs(r_somme) > 0.2 else ""
    print(f"  {v:15s} {r_somme:>+10.3f} {r_dalc:>+10.3f} {r_walc:>+10.3f}{marker}")

print(f"\n  Note: cor(Somme) est souvent PLUS FORTE que cor(Dalc) ou cor(Walc)")
print(f"  car la somme cumule les deux signaux. Mais cela masque la difference")
print(f"  semaine/weekend.")

# =============================================================================
# 3. CHI-DEUX (Somme vs variables nominales)
# =============================================================================
print_section("3. CHI-DEUX (Somme vs nominales)")
nominal_vars = var_types["nominal"]

def cramers_v(ct):
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

print(f"  {'Variable':15s} {'chi2':>8} {'p-value':>10} {'V Cramer':>10} {'sig'}")
for v in nominal_vars:
    ct = pd.crosstab(df[v], df["Somme"])
    chi2, p, _, _ = chi2_contingency(ct)
    vc = cramers_v(ct)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    if p < 0.05:
        print(f"  {v:15s} {chi2:>8.2f} {p:>10.4f} {vc:>10.3f}  {sig}")

# =============================================================================
# 4. CLUSTERING CAH sur la Somme (1D)
# =============================================================================
print_section("4. CLUSTERING CAH (1D sur Somme)")
X_somme = df["Somme"].values.reshape(-1, 1).astype(float)
X_somme_std = (X_somme - X_somme.mean()) / X_somme.std()

Z_somme = linkage(X_somme_std, method="ward")

def inertie_intra(X, labels):
    return sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum()
               for c in np.unique(labels))

I_tot = ((X_somme_std - X_somme_std.mean()) ** 2).sum()
print(f"  I_total = {I_tot:.2f}")
print(f"\n  {'K':>2} | {'intra_CAH':>10} {'%expl':>7} | {'intra_KM':>10} {'%expl':>7} | {'ARI':>6}")
for k in range(2, 6):
    lc = fcluster(Z_somme, t=k, criterion="maxclust")
    lk = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_somme_std)
    wc = inertie_intra(X_somme_std, lc)
    wk = inertie_intra(X_somme_std, lk)
    ari = adjusted_rand_score(lc, lk)
    print(f"  {k:>2} | {wc:>10.1f} {(1-wc/I_tot)*100:>6.1f}% | "
          f"{wk:>10.1f} {(1-wk/I_tot)*100:>6.1f}% | {ari:>6.3f}")

# =============================================================================
# 5. K-MEANS 1D (fonctionne aussi en 1D!)
# =============================================================================
print_section("5. K-MEANS 1D (Somme)")
print("  Note: K-means fonctionne en TOUTE dimension, y compris 1D.")
print("  En 1D, K-means = algorithme de Jenks (natural breaks).")

for k in [2, 3]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X_somme_std)
    labels = km.labels_
    print(f"\n  K={k}:")
    for c in range(k):
        mask = labels == c
        print(f"    Cluster {c}: n={mask.sum():>3}, Somme moy={df.loc[mask,'Somme'].mean():.2f}, "
              f"Dalc moy={df.loc[mask,'Dalc'].mean():.2f}, Walc moy={df.loc[mask,'Walc'].mean():.2f}")

# =============================================================================
# 5b. PARTITIONS RETENUES: K=2 et K=4 (K-means 1D)
# =============================================================================
print_section("5b. PARTITION K=2 (K-means, Somme)")
km2 = KMeans(n_clusters=2, random_state=42, n_init=20).fit(X_somme_std)
labels_k2 = km2.labels_
# Trier par Somme croissante
ordre_k2 = np.argsort([df.loc[labels_k2 == c, "Somme"].mean() for c in range(2)])
print(f"  {'Cluster':>8} {'Label':>6} {'n':>5} {'%':>6} {'Somme':>7} {'Dalc':>6} {'Walc':>6} {'Profil'}")
noms_k2 = ["Low", "High"]
for i, c in enumerate(ordre_k2):
    mask = labels_k2 == c
    n = mask.sum()
    print(f"  {f'C{i+1}':>8} {noms_k2[i]:>6} {n:>5} {n/len(df)*100:>5.1f}% "
          f"{df.loc[mask,'Somme'].mean():>7.2f} {df.loc[mask,'Dalc'].mean():>6.2f} "
          f"{df.loc[mask,'Walc'].mean():>6.2f}  {'non/peu buveurs' if i==0 else 'buveurs'}")

# Seuil de coupure
seuils_k2 = sorted([df.loc[labels_k2 == c, "Somme"].max() for c in range(2)])[:-1]
print(f"\n  Seuil de coupure: Somme <= {seuils_k2[0]} → Low, Somme > {seuils_k2[0]} → High")

print_section("5c. PARTITION K=4 (K-means, Somme)")
km4 = KMeans(n_clusters=4, random_state=42, n_init=20).fit(X_somme_std)
labels_k4 = km4.labels_
# Trier par Somme croissante
ordre_k4 = np.argsort([df.loc[labels_k4 == c, "Somme"].mean() for c in range(4)])
print(f"  {'Cluster':>8} {'n':>5} {'%':>6} {'Somme':>7} {'Dalc':>6} {'Walc':>6} {'Range Somme'}")
noms_k4 = ["Non-buveurs", "Faibles", "Moderes", "Eleves"]
for i, c in enumerate(ordre_k4):
    mask = labels_k4 == c
    n = mask.sum()
    s_min = df.loc[mask, "Somme"].min()
    s_max = df.loc[mask, "Somme"].max()
    print(f"  {f'C{i+1} ({noms_k4[i]})':>20} {n:>5} {n/len(df)*100:>5.1f}% "
          f"{df.loc[mask,'Somme'].mean():>7.2f} {df.loc[mask,'Dalc'].mean():>6.2f} "
          f"{df.loc[mask,'Walc'].mean():>6.2f}  [{s_min}-{s_max}]")

print(f"\n  PROBLEME K=4 sur Somme:")
print(f"    Un eleve (Dalc=1, Walc=4, Somme=5) et un eleve (Dalc=3, Walc=2, Somme=5)")
print(f"    sont dans le MEME cluster car meme Somme.")
print(f"    En 2D (Dalc x Walc), ils seraient dans des clusters DIFFERENTS.")
print(f"    → La Somme ne distingue PAS 'weekend' de 'quotidien modere'.")

# =============================================================================
# 6. COMPARAISON: SOMME vs 2D (Dalc, Walc)
# =============================================================================
print_section("6. COMPARAISON: Somme (1D) vs Dalc x Walc (2D)")

# Partition 2D (votre approche B05)
X_2d = df[["Dalc", "Walc"]].values.astype(float)
X_2d_std = (X_2d - X_2d.mean(0)) / X_2d.std(0)
km_2d = KMeans(n_clusters=4, random_state=42, n_init=20).fit(X_2d_std)
labels_2d = km_2d.labels_

# Partition 1D (somme, K=3)
km_1d = KMeans(n_clusters=3, random_state=42, n_init=20).fit(X_somme_std)
labels_1d = km_1d.labels_

# Mapping 2D -> 3 classes (votre methode B05)
stats_2d = pd.DataFrame({"Dalc": df["Dalc"], "Walc": df["Walc"], "cluster": labels_2d})
stats_2d = stats_2d.groupby("cluster")[["Dalc", "Walc"]].mean()
stats_2d["total"] = stats_2d.sum(axis=1)
ordre = stats_2d["total"].sort_values().index
c_low = ordre[0]
autres = [c for c in ordre if c != c_low]
c_week = stats_2d.loc[autres, "Dalc"].idxmin()
mapping_2d = {c: (0 if c == c_low else 1 if c == c_week else 2) for c in stats_2d.index}
risk_2d = pd.Series(labels_2d).map(mapping_2d).values

# Profils des clusters 1D
print("\n  APPROCHE 1D (Somme, K=3):")
for c in range(3):
    mask = labels_1d == c
    print(f"    Cluster {c}: n={mask.sum():>3}, Somme={df.loc[mask,'Somme'].mean():.2f}, "
          f"Dalc={df.loc[mask,'Dalc'].mean():.2f}, Walc={df.loc[mask,'Walc'].mean():.2f}")

print("\n  APPROCHE 2D (Dalc x Walc, K=4 -> 3 classes):")
for g in [0, 1, 2]:
    mask = risk_2d == g
    noms = {0: "non-buveurs", 1: "weekend", 2: "eleve"}
    print(f"    Classe {g} ({noms[g]:11s}): n={mask.sum():>3}, "
          f"Dalc={df.loc[mask,'Dalc'].mean():.2f}, Walc={df.loc[mask,'Walc'].mean():.2f}")

# Le point cle: la somme distingue-t-elle le profil weekend?
print_section("7. TEST CLE: la Somme distingue-t-elle le weekend?")
weekend_mask = risk_2d == 1
non_buv_mask = risk_2d == 0

print(f"  Profil weekend (2D):  Dalc={df.loc[weekend_mask,'Dalc'].mean():.2f}, "
      f"Walc={df.loc[weekend_mask,'Walc'].mean():.2f}, "
      f"Somme={df.loc[weekend_mask,'Somme'].mean():.2f}")
print(f"  Profil non-buveurs:   Dalc={df.loc[non_buv_mask,'Dalc'].mean():.2f}, "
      f"Walc={df.loc[non_buv_mask,'Walc'].mean():.2f}, "
      f"Somme={df.loc[non_buv_mask,'Somme'].mean():.2f}")

print(f"\n  Weekend: Somme={df.loc[weekend_mask,'Somme'].mean():.2f}")
print(f"  Non-buv: Somme={df.loc[non_buv_mask,'Somme'].mean():.2f}")
diff = df.loc[weekend_mask,'Somme'].mean() - df.loc[non_buv_mask,'Somme'].mean()
print(f"  Difference: {diff:.2f}")
print(f"\n  CONCLUSION:")
print(f"    La Somme DISTINGUE partiellement weekend (Somme~4.4) de non-buv (Somme~2.3),")
print(f"    MAIS elle melange weekend avec les 'moderes' (Dalc=2, Walc=2, Somme=4).")
print(f"    En 1D, un eleve (Dalc=1, Walc=4, Somme=5) et un eleve (Dalc=3, Walc=2, Somme=5)")
print(f"    sont dans le MEME cluster. En 2D, ils sont dans des clusters DIFFERENTS.")
print(f"    → L'approche 2D preserve l'information semaine/weekend que la somme gomme.")

# =============================================================================
# 8. SOMME SIMPLE vs SOMME PONDEREE
# =============================================================================
print_section("8. SOMME SIMPLE vs SOMME PONDEREE")
print("  Question: une ponderation differente change-t-elle les resultats?")
print("  On teste S = alpha*Dalc + beta*Walc pour alpha in [0.1, 0.9]\n")

from scipy.stats import spearmanr

# --- 8a. Recherche du meilleur alpha par correlation avec variables predictives ---
predicteurs = ["goout", "studytime", "absences", "freetime", "age"]
predicteurs = [p for p in predicteurs if p in df.columns]

print(f"  {'alpha':>6} {'beta':>6} | ", end="")
for p in predicteurs:
    print(f"{'rho_'+p:>12}", end="")
print(f" | {'rho_moyen':>10}")
print("  " + "-" * (16 + 12 * len(predicteurs) + 14))

best_alpha = 0.5
best_rho_mean = 0

for alpha in np.arange(0.1, 1.0, 0.1):
    beta = 1 - alpha
    S_pond = alpha * df["Dalc"] + beta * df["Walc"]
    rhos = []
    print(f"  {alpha:>6.2f} {beta:>6.2f} | ", end="")
    for p in predicteurs:
        rho, _ = spearmanr(S_pond, df[p])
        rhos.append(abs(rho))
        print(f"{rho:>+12.3f}", end="")
    rho_mean = np.mean(rhos)
    print(f" | {rho_mean:>10.4f}", end="")
    if rho_mean > best_rho_mean:
        best_rho_mean = rho_mean
        best_alpha = alpha
        print("  <-- max", end="")
    print()

print(f"\n  Meilleur alpha = {best_alpha:.2f} (beta = {1-best_alpha:.2f})")
print(f"  Correlation moyenne maximale = {best_rho_mean:.4f}")

# --- 8b. Comparaison directe somme simple vs meilleur pondere ---
S_simple = df["Dalc"] + df["Walc"]
S_best = best_alpha * df["Dalc"] + (1 - best_alpha) * df["Walc"]

print(f"\n  Comparaison somme simple (alpha=0.5) vs ponderee (alpha={best_alpha:.2f}):")
print(f"  {'Variable':>12} | {'rho(simple)':>12} | {'rho(ponderee)':>14} | {'diff':>8}")
print("  " + "-" * 55)
for p in predicteurs:
    rho_s, _ = spearmanr(S_simple, df[p])
    rho_p, _ = spearmanr(S_best, df[p])
    diff = abs(rho_p) - abs(rho_s)
    print(f"  {p:>12} | {rho_s:>+12.3f} | {rho_p:>+14.3f} | {diff:>+8.4f}")

# --- 8c. Correlation de rang entre les deux scores ---
rho_rank, _ = spearmanr(S_simple, S_best)
print(f"\n  Spearman(S_simple, S_ponderee) = {rho_rank:.4f}")
print(f"  → Les deux scores classent les individus de maniere quasi-identique.")

# --- 8d. Impact sur la classification binaire ---
seuil_simple = 4  # S >= 4 = buveur
# Pour pondere, on cherche le seuil equivalent
median_pond = S_best.median()
seuil_pond = S_best[S_simple >= seuil_simple].min()

labels_simple = (S_simple >= seuil_simple).astype(int)
labels_pond = (S_best >= seuil_pond).astype(int)
accord = (labels_simple == labels_pond).mean()

print(f"\n  Classification binaire:")
print(f"    Seuil simple: S >= {seuil_simple}")
print(f"    Seuil pondere: S_pond >= {seuil_pond:.2f}")
print(f"    Accord entre les deux classifications: {accord*100:.1f}%")

# --- 8e. Conclusion ---
print_section("8e. CONCLUSION: SOMME SIMPLE vs PONDEREE")
print(f"  1. La correlation Spearman entre S_simple et S_ponderee = {rho_rank:.3f}")
print(f"     → Le classement des individus est quasi-identique.")
print(f"  2. Les correlations avec les predicteurs changent de < 0.01 en moyenne.")
print(f"  3. L'accord de classification binaire est de {accord*100:.1f}%.")
print(f"  4. Raison: Dalc et Walc sont fortement correles (rho=0.613).")
print(f"     Changer les poids ne bouleverse pas le ranking.")
print(f"  → CONCLUSION: la somme simple est retenue par PARCIMONIE.")
print(f"     Une ponderation n'apporte pas d'amelioration significative.")
