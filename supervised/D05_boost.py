"""
D05 — METHODE DU BOOTSTRAP: estimation du taux d'erreur et intervalle de confiance.
      Inclut tous les modeles (LDA, Logistique, KNN, RF, GB) pour comparaison finale.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_section, save_plot, BASE_DIR

print_section("D05 — METHODE DU BOOTSTRAP")
X_train = np.loadtxt(os.path.join(BASE_DIR, "X_train.csv"), delimiter=",")
X_test = np.loadtxt(os.path.join(BASE_DIR, "X_test.csv"), delimiter=",")
y_train = np.loadtxt(os.path.join(BASE_DIR, "y_train.csv"), delimiter=",").astype(int)
y_test = np.loadtxt(os.path.join(BASE_DIR, "y_test.csv"), delimiter=",").astype(int)
X = np.vstack([X_train, X_test])
y = np.concatenate([y_train, y_test])
n = X.shape[0]
print(f"  n={n}, baseline={np.bincount(y).max()/len(y):.4f}")

# K=33 (coherent avec D01 GridSearchCV)
modeles = {
    "LDA": LinearDiscriminantAnalysis(),
    "Logistique": LogisticRegression(max_iter=2000, random_state=42),
    "KNN (K=15)": KNeighborsClassifier(n_neighbors=15),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5,
                                            max_features="sqrt", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                     learning_rate=0.1, random_state=42),
}

# Bootstrap: B tirages avec remise. On entraine sur l'echantillon bootstrap,
# on teste sur les points out-of-bag (non tires, ~36.8%).
B = 300
rng = np.random.default_rng(42)
print_section(f"BOOTSTRAP ({B} re-echantillonnages)")
print(f"  {'Modele':20s} {'acc_OOB':>9} {'IC 95%':>20} {'acc_.632':>10}")

resultats = {}
for nom, mod in modeles.items():
    acc_oob, acc_632 = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) == 0:
            continue
        try:
            m = mod.__class__(**mod.get_params()).fit(X[idx], y[idx])
        except Exception:
            m = mod.fit(X[idx], y[idx])
        a_oob = accuracy_score(y[oob], m.predict(X[oob]))
        a_train = accuracy_score(y[idx], m.predict(X[idx]))
        acc_oob.append(a_oob)
        acc_632.append(0.368 * a_train + 0.632 * a_oob)
    acc_oob = np.array(acc_oob)
    ic = np.percentile(acc_oob, [2.5, 97.5])
    resultats[nom] = acc_oob
    print(f"  {nom:20s} {acc_oob.mean():>9.4f} [{ic[0]:.4f}, {ic[1]:.4f}] {np.mean(acc_632):>10.4f}")

print("\n  acc_OOB  : moyenne des accuracy sur les points out-of-bag")
print("  IC 95%   : intervalle de confiance (percentiles 2.5 et 97.5)")
print("  acc_.632 : estimateur .632 = 0.368*train + 0.632*OOB")

# Comparaison des IC: chevauchement?
print_section("COMPARAISON DES INTERVALLES DE CONFIANCE")
for nom, acc in resultats.items():
    ic = np.percentile(acc, [2.5, 97.5])
    print(f"  {nom:20s}: [{ic[0]:.4f}, {ic[1]:.4f}]")
print("\n  Si les IC se chevauchent, les modeles ne sont pas statistiquement")
print("  differents au seuil 5%.")

# Distribution bootstrap des accuracy par modele
plt.figure(figsize=(12, 5))
for nom, acc in resultats.items():
    plt.hist(acc, bins=30, alpha=0.4, label=f"{nom} (m={acc.mean():.3f})")
plt.axvline(np.bincount(y).max()/len(y), color="black", ls="--", label="baseline")
plt.xlabel("accuracy (out-of-bag)"); plt.ylabel("frequence")
plt.title(f"Distribution bootstrap de l'accuracy ({B} re-echantillonnages)")
plt.legend(fontsize=8)
plt.tight_layout()
save_plot("D05_bootstrap_distribution.png")
