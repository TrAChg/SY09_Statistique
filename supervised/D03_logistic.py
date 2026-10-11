"""
D03 — REGRESSION LOGISTIQUE MULTINOMIALE: coefficients, regle de cout, comparaison.
     Inclut justification de la matrice de cout et analyse de sensibilite.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_section, save_plot, BASE_DIR

print_section("D03 — REGRESSION LOGISTIQUE MULTINOMIALE")
X_train = np.loadtxt(os.path.join(BASE_DIR, "X_train.csv"), delimiter=",")
X_test = np.loadtxt(os.path.join(BASE_DIR, "X_test.csv"), delimiter=",")
y_train = np.loadtxt(os.path.join(BASE_DIR, "y_train.csv"), delimiter=",").astype(int)
y_test = np.loadtxt(os.path.join(BASE_DIR, "y_test.csv"), delimiter=",").astype(int)
feature_names = pd.read_csv(os.path.join(BASE_DIR, "feature_names.csv")).iloc[:, 0].tolist()
classes = np.unique(y_train)
LABELS = ["non-buveurs", "weekend", "eleve"]
baseline = np.bincount(y_test).max() / len(y_test)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}, classes={list(classes)}")
print(f"  Baseline: {baseline:.4f}")

# Modele multinomial
print_section("APPRENTISSAGE")
model = LogisticRegression(max_iter=2000, random_state=42).fit(X_train, y_train)
acc_train = accuracy_score(y_train, model.predict(X_train))
acc_test = accuracy_score(y_test, model.predict(X_test))
print(f"  Train: {acc_train:.4f} | Test: {acc_test:.4f}")

# Coefficients par classe
print_section("COEFFICIENTS PAR CLASSE")
coef = pd.DataFrame(model.coef_.T, index=feature_names, columns=LABELS)
print(coef.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, lab in zip(axes, LABELS):
    s = coef[lab].sort_values()
    colors = ["coral" if v > 0 else "steelblue" for v in s]
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="black", lw=1); ax.set_title(f"classe {lab}")
plt.tight_layout()
save_plot("D03_logistic_coefficients.png")

# =============================================================================
# REGLE MINIMISANT LE RISQUE (matrice de cout, poly ch.9)
#
# JUSTIFICATION DE LA MATRICE DE COUT:
# Dans un contexte de PREVENTION scolaire (detection d'eleves a risque):
#   - Sous-estimer le risque (predire "non-buveur" pour un "eleve") est GRAVE:
#     l'eleve ne recoit pas l'aide necessaire. Cout = 3 (le plus eleve).
#   - Sur-estimer le risque (predire "eleve" pour un "non-buveur") est moins
#     grave: l'eleve recoit une attention supplementaire non necessaire. Cout = 1.
#   - Confondre "weekend" et "non-buveur" (sous-estimation moderee): cout = 2.
#   - Confondre "weekend" et "eleve" (sur-estimation moderee): cout = 1.
#
# Principe: C(k, a) = cout quand la vraie classe est k et on predit a.
# Les couts sont ORDINAUX (rapport entre eux compte, pas la valeur absolue).
# =============================================================================
print_section("REGLE DE COUT (matrice 3x3)")
C = np.array([[0, 1, 1],   # vrai non-buveurs: erreur symetrique (pas grave)
              [2, 0, 1],   # vrai weekend: sous-estimer (->non-buv) coute 2x
              [3, 2, 0]])  # vrai eleve: sous-estimer coute 3x (le plus grave)
print("  Matrice de cout C(vrai, predit):")
print("  JUSTIFICATION: prevention scolaire -> sous-estimer le risque est")
print("  plus grave que sur-estimer (mieux vaut alerter a tort que manquer un cas)")
print("            " + " ".join(f"{l:>11s}" for l in LABELS))
for i, l in enumerate(LABELS):
    print(f"  {l:11s} " + " ".join(f"{C[i,j]:>11d}" for j in range(3)))

proba = model.predict_proba(X_test)
risque = proba @ C
pred_cout = classes[risque.argmin(axis=1)]
pred_map = model.predict(X_test)

def cout_total(preds):
    return sum(C[t, p] for t, p in zip(y_test, preds))

print(f"\n  {'regle':18s} {'accuracy':>9} {'cout total':>11} {'rappel eleve':>13}")
for name, preds in [("MAP (argmax P)", pred_map), ("Cout minimal", pred_cout)]:
    rec_eleve = recall_score(y_test, preds, average=None, labels=classes)[2]
    print(f"  {name:18s} {accuracy_score(y_test, preds):>9.4f} {cout_total(preds):>11d} {rec_eleve:>13.3f}")

# =============================================================================
# ANALYSE DE SENSIBILITE DE LA MATRICE DE COUT
# On varie le cout de sous-estimation de la classe "eleve" (C[2,0]) de 1 a 5
# pour montrer que la conclusion est ROBUSTE: quel que soit le cout exact,
# la regle de cout detecte mieux la classe "eleve" que la regle MAP.
# =============================================================================
print_section("ANALYSE DE SENSIBILITE (cout sous-estimation 'eleve')")
print(f"  {'C(eleve,non-buv)':>18s} {'accuracy':>9} {'cout total':>11} {'rappel eleve':>13} {'rappel weekend':>14}")
for cost_high in [1, 2, 3, 4, 5]:
    C_var = np.array([[0, 1, 1],
                      [2, 0, 1],
                      [cost_high, 2, 0]])
    risque_var = proba @ C_var
    pred_var = classes[risque_var.argmin(axis=1)]
    acc_var = accuracy_score(y_test, pred_var)
    cout_var = sum(C_var[t, p] for t, p in zip(y_test, pred_var))
    rec_var = recall_score(y_test, pred_var, average=None, labels=classes)
    marker = " <-- retenu" if cost_high == 3 else ""
    print(f"  {cost_high:>18d} {acc_var:>9.4f} {cout_var:>11d} {rec_var[2]:>13.3f} {rec_var[1]:>14.3f}{marker}")

print("\n  CONCLUSION: le rappel 'eleve' augmente avec le cout de sous-estimation,")
print("  tandis que le weekend reste quasi-indetectable quelle que soit la matrice.")
print("  La matrice retenue (C[2,0]=3) offre un bon compromis rappel/accuracy.")

# Comparaison finale
print_section("COMPARAISON FINALE")
lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
best_k = max(range(1, 51),
             key=lambda k: cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5).mean())
knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
for name, m in [("Baseline", None), (f"KNN (K={best_k})", knn), ("LDA", lda), ("Logistique", model)]:
    acc = baseline if m is None else accuracy_score(y_test, m.predict(X_test))
    print(f"  {name:15s}: {acc:.4f}")

# Validation croisee + rappel par classe
print_section("VALIDATION CROISEE (Stratified 5-fold)")
X_full = np.vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in [("Logistique", LogisticRegression(max_iter=2000, random_state=42)),
                ("LDA", LinearDiscriminantAnalysis()),
                (f"KNN (K={best_k})", KNeighborsClassifier(n_neighbors=best_k))]:
    scores = cross_val_score(m, X_full, y_full, cv=skf)
    print(f"  {name:15s}: {scores.mean():.4f} +/- {scores.std():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, m) in zip(axes, [("Logistique", model), ("LDA", lda), ("KNN", knn)]):
    cm = confusion_matrix(y_test, m.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} — Acc={accuracy_score(y_test, m.predict(X_test)):.3f}")
plt.tight_layout()
save_plot("D03_final_comparison.png")
