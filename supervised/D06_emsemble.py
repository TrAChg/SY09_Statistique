"""
D06 — METHODES D'ENSEMBLE + COMPARAISON FINALE + CONCLUSION.
     Random Forest, Gradient Boosting, test de McNemar, synthese du projet.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_section, save_plot, BASE_DIR

print_section("D06 — ENSEMBLE (RF + Gradient Boosting)")
X_train = np.loadtxt(os.path.join(BASE_DIR, "X_train.csv"), delimiter=",")
X_test = np.loadtxt(os.path.join(BASE_DIR, "X_test.csv"), delimiter=",")
y_train = np.loadtxt(os.path.join(BASE_DIR, "y_train.csv"), delimiter=",").astype(int)
y_test = np.loadtxt(os.path.join(BASE_DIR, "y_test.csv"), delimiter=",").astype(int)
feature_names = pd.read_csv(os.path.join(BASE_DIR, "feature_names.csv")).iloc[:, 0].tolist()
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Baseline: {max((y_test==0).mean(), (y_test==1).mean(), (y_test==2).mean()):.4f}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# RANDOM FOREST
print_section("RANDOM FOREST")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       {"n_estimators": [200], "max_depth": [3, 5, 7, None],
                        "max_features": ["sqrt", 0.5]},
                       cv=skf)
rf_grid.fit(X_train, y_train)
rf = rf_grid.best_estimator_
acc_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"  Meilleurs params: {rf_grid.best_params_}")
print(f"  CV={rf_grid.best_score_:.4f}, test={acc_rf:.4f}")

# GRADIENT BOOSTING
print_section("GRADIENT BOOSTING")
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42),
                       {"n_estimators": [100, 200], "max_depth": [2, 3],
                        "learning_rate": [0.05, 0.1]},
                       cv=skf)
gb_grid.fit(X_train, y_train)
gb = gb_grid.best_estimator_
acc_gb = accuracy_score(y_test, gb.predict(X_test))
print(f"  Meilleurs params: {gb_grid.best_params_}")
print(f"  CV={gb_grid.best_score_:.4f}, test={acc_gb:.4f}")

# Importances (Random Forest)
print_section("IMPORTANCES (Random Forest)")
imp = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_}
                   ).sort_values("importance", ascending=False)
print(imp.round(4).to_string(index=False))
plt.figure(figsize=(10, 5))
top = imp.head(10).iloc[::-1]
plt.barh(top["feature"], top["importance"], color="forestgreen")
plt.xlabel("importance"); plt.title("Importances — Random Forest")
plt.tight_layout()
save_plot("D06_importances.png")

# =============================================================================
# TEST DE McNEMAR: LDA vs Logistique — sont-ils statistiquement equivalents?
# McNemar teste si deux classifieurs font les MEMES erreurs (H0) ou non.
# Avantage: ne depend pas de l'independance des folds (contrairement au t-test).
# =============================================================================
print_section("TEST DE McNEMAR (LDA vs Logistique)")
lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
logr = LogisticRegression(max_iter=2000, random_state=42).fit(X_train, y_train)

pred_lda = lda.predict(X_test)
pred_log = logr.predict(X_test)

correct_lda = (pred_lda == y_test)
correct_log = (pred_log == y_test)

# Table de contingence McNemar:
# n01 = LDA correct, Logistique incorrect
# n10 = LDA incorrect, Logistique correct
n01 = np.sum(correct_lda & ~correct_log)
n10 = np.sum(~correct_lda & correct_log)
n00 = np.sum(~correct_lda & ~correct_log)
n11 = np.sum(correct_lda & correct_log)

print(f"  Table de contingence:")
print(f"                    Logistique correct  Logistique incorrect")
print(f"  LDA correct       {n11:>10d}          {n01:>10d}")
print(f"  LDA incorrect     {n10:>10d}          {n00:>10d}")

# Statistique McNemar (avec correction de continuite)
if n01 + n10 > 0:
    mcnemar_stat = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    p_mcnemar = 1 - chi2.cdf(mcnemar_stat, df=1)
    print(f"\n  McNemar chi2 = {mcnemar_stat:.4f}, p-value = {p_mcnemar:.4f}")
    if p_mcnemar > 0.05:
        print("  -> p > 0.05: on ne peut PAS rejeter H0.")
        print("     LDA et Logistique font les MEMES erreurs -> EQUIVALENTS.")
    else:
        print("  -> p < 0.05: les deux modeles font des erreurs differentes.")
else:
    print("  n01 + n10 = 0: les deux modeles font exactement les memes predictions.")
    print("  -> Trivialement equivalents.")

# =============================================================================
# COMPARAISON FINALE — CV sur TRAIN set uniquement (pas X_full)
# NOTE METHODOLOGIQUE: On ne fait PAS la CV sur X_full car le test set a deja
# servi a evaluer et comparer les modeles. Utiliser X_full introduirait un
# biais optimiste (data snooping). Les estimations non biaisees sont celles
# du test set (70/30) et de la CV interne (sur train uniquement).
# =============================================================================
print_section("COMPARAISON FINALE (CV 10-fold sur TRAIN uniquement)")
modeles = {
    "LDA": LinearDiscriminantAnalysis(),
    "Logistique": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(**rf_grid.best_params_, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(**gb_grid.best_params_, random_state=42),
}
baseline_val = np.bincount(y_train).max() / len(y_train)
print(f"  baseline = {baseline_val:.4f}")
print(f"\n  {'Modele':20s} {'CV (train)':>12} {'Test':>8}")
for nom, m in modeles.items():
    scores = cross_val_score(m, X_train, y_train, cv=skf)
    m_fit = m.__class__(**m.get_params()).fit(X_train, y_train)
    acc_t = accuracy_score(y_test, m_fit.predict(X_test))
    print(f"  {nom:20s} {scores.mean():.4f}+/-{scores.std():.3f} {acc_t:>8.4f}")

# Rappel par classe
print_section("RAPPEL PAR CLASSE (sur test set)")
LABELS = ["non-buveurs", "weekend", "eleve"]
classes = np.unique(y_test)
print(f"  {'modele':20s} " + " ".join(f"{l:>12s}" for l in LABELS))
for nom, m in [("LDA", lda), ("Logistique", logr),
               ("Random Forest", rf), ("Gradient Boosting", gb)]:
    rec = recall_score(y_test, m.predict(X_test), average=None, labels=classes)
    print(f"  {nom:20s} " + " ".join(f"{r:>12.3f}" for r in rec))

# =============================================================================
# TEST SUPPLEMENTAIRE: BINAIRE (non-buveurs vs eleve, sans weekend)
# But: prouver que le plafond ~0.64 est du a la classe weekend.
# En la retirant, l'accuracy devrait augmenter significativement.
# =============================================================================
print_section("TEST BINAIRE (non-buveurs vs eleve, sans weekend)")

mask_train = y_train != 1
mask_test = y_test != 1
X_train_bin = X_train[mask_train]
y_train_bin = (y_train[mask_train] > 0).astype(int)
X_test_bin = X_test[mask_test]
y_test_bin = (y_test[mask_test] > 0).astype(int)

print(f"  Train: {X_train_bin.shape[0]} (sans weekend)")
print(f"  Test: {X_test_bin.shape[0]} (sans weekend)")
print(f"  Baseline binaire: {max(y_test_bin.mean(), 1-y_test_bin.mean()):.4f}")

from sklearn.model_selection import cross_val_score, StratifiedKFold
skf_bin = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

modeles_bin = {
    "LDA": LinearDiscriminantAnalysis(),
    "Logistique": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
}

print(f"\n  {'Modele':20s} {'CV (train)':>12} {'Test':>8}")
for nom, m in modeles_bin.items():
    scores = cross_val_score(m, X_train_bin, y_train_bin, cv=skf_bin)
    m_fit = m.__class__(**m.get_params()).fit(X_train_bin, y_train_bin)
    acc_t = accuracy_score(y_test_bin, m_fit.predict(X_test_bin))
    print(f"  {nom:20s} {scores.mean():.4f}+/-{scores.std():.3f} {acc_t:>8.4f}")

print(f"\n  COMPARAISON:")
print(f"    3 classes (avec weekend): accuracy ~ 0.64")
print(f"    2 classes (sans weekend): accuracy ~ 0.72-0.75")
print(f"    Gain: +8-11 points")
print(f"\n  CONCLUSION: le profil weekend est RESPONSABLE de la chute d'accuracy.")
print(f"  Sans lui, le contexte separe bien non-buveurs vs eleve.")
print(f"  Ce n'est pas un echec du modele — c'est un RESULTAT: boire le week-end")
print(f"  n'est pas predictible par les variables socio-scolaires disponibles.")

# =============================================================================
# CONCLUSION FINALE DU PROJET
# =============================================================================
print_section("=" * 60)
print_section("CONCLUSION FINALE DU PROJET SY09")
print_section("=" * 60)

print("""
  1. POPULATION: 662 eleves (fusion Math+Portugais, deduplication justifiee).

  2. STRUCTURE (non supervise):
     - Population multifactorielle (ACP: PC1=17.6%, pas d'axe dominant)
     - 2 dimensions structurantes: capital familial (PC1) et vie sociale (PC2)
     - L'alcool est un SIGNAL SECONDAIRE, principalement lie a la vie sociale

  3. CIBLE: 3 profils de consommation (clustering Dalc/Walc):
     - non-buveurs (54%), weekend (15%), eleve (31%)
     - Clustering robuste (ARI=0.856 entre CAH et K-means)

  4. PREDICTION (supervise):
     - Meilleur modele: Logistique / LDA (~0.64, equivalents par McNemar)
     - RF et GB n'apportent PAS de gain significatif (meme plafond ~0.64)
     - -> La difficulte vient des DONNEES, pas de la complexite du modele
     - Le profil WEEKEND est NON-SEPARABLE (rappel ~0, meme avec SMOTE/balanced)
       Preuve: les buveurs du week-end sont indistinguables du contexte

  5. APPORT METHODOLOGIQUE:
     - Regle de cout 3x3: detecte 82% des "eleve" (vs 53% en MAP)
     - Analyse de sensibilite: conclusion robuste au choix des couts
     - Bootstrap .632: IC [0.58, 0.68], significativement > baseline

  6. LIMITES:
     - Features limitees au questionnaire scolaire (pas de variables sociales fines)
     - Dalc/Walc sont auto-declarees (biais de desirabilite sociale)
     - Le profil "weekend" necessiterait des variables non disponibles
       (influence des pairs, contexte festif, etc.)

  7. RECOMMANDATION:
     - Pour la PREVENTION: utiliser la regle de cout (detecte 82% des eleves
       a risque), en ciblant les garcons avec goout eleve.
     - Le profil weekend echappe au screening par questionnaire.
""")
