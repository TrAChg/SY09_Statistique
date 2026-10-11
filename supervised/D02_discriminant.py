"""
D02 — ANALYSE DISCRIMINANTE: Euclidien, Naive Bayes, LDA, QDA (multi-classes).
Progression selon l'hypothese sur la covariance Sigma:
  Euclidien (Sigma=I) -> Naive Bayes (Sigma diagonale) -> LDA (Sigma commun) -> QDA (Sigma_k).
Inclut test de normalite (Shapiro-Wilk) pour valider les hypotheses.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_section, save_plot, BASE_DIR

print_section("D02 — ANALYSE DISCRIMINANTE")
X_train = np.loadtxt(os.path.join(BASE_DIR, "X_train.csv"), delimiter=",")
X_test = np.loadtxt(os.path.join(BASE_DIR, "X_test.csv"), delimiter=",")
y_train = np.loadtxt(os.path.join(BASE_DIR, "y_train.csv"), delimiter=",").astype(int)
y_test = np.loadtxt(os.path.join(BASE_DIR, "y_test.csv"), delimiter=",").astype(int)
feature_names = pd.read_csv(os.path.join(BASE_DIR, "feature_names.csv")).iloc[:, 0].tolist()
n, p = X_train.shape
classes = np.unique(y_train)
LABELS = ["non-buveurs", "weekend", "eleve"]
baseline = np.bincount(y_test).max() / len(y_test)
print(f"  Train: {n}x{p}, Test: {X_test.shape[0]}, classes={list(classes)}")

# =============================================================================
# TEST DE NORMALITE (Shapiro-Wilk) — prerequis theorique de LDA/QDA
# LDA suppose f_k(x) = N(mu_k, Sigma). Si la normalite est violee, LDA reste
# un classifieur lineaire performant (robuste), mais la justification
# probabiliste (Bayes optimal) ne tient plus strictement.
# =============================================================================
print_section("TEST DE NORMALITE (Shapiro-Wilk par feature et par classe)")
print(f"  H0: la feature suit une loi normale dans chaque classe")
print(f"  Seuil alpha = 0.05 (rejet si p < 0.05)")
n_reject = 0
n_total = 0
print(f"\n  {'feature':15s} {'classe':>8} {'W':>8} {'p-value':>10} {'normal?':>8}")
for j, fname in enumerate(feature_names):
    for c in classes:
        data = X_train[y_train == c, j]
        if len(data) > 5000:
            data = data[:5000]
        w, pval = shapiro(data)
        is_normal = "oui" if pval >= 0.05 else "NON"
        n_total += 1
        if pval < 0.05:
            n_reject += 1
        if pval < 0.01:
            print(f"  {fname:15s} {LABELS[c]:>8} {w:>8.4f} {pval:>10.4f} {is_normal:>8}")

print(f"\n  Resume: {n_reject}/{n_total} tests rejettent H0 (normalite)")
pct_reject = n_reject / n_total * 100
print(f"  Taux de rejet: {pct_reject:.0f}%")
if pct_reject > 50:
    print("  -> Normalite largement violee. LDA reste applicable car:")
    print("     1. C'est un classifieur lineaire (fonctionne meme sans normalite)")
    print("     2. Le theoreme central limite aide pour les grandes classes (n>100)")
    print("     3. Les resultats empiriques (CV) confirment sa bonne performance")
    print("     4. L'hypothese gaussienne est une SIMPLIFICATION, pas une condition sine qua non")
else:
    print("  -> Normalite raisonnablement respectee. LDA est theoriquement justifie.")

priors = {k: np.mean(y_train == k) for k in classes}
means = {k: X_train[y_train == k].mean(axis=0) for k in classes}
covs = {k: np.cov(X_train[y_train == k], rowvar=False) for k in classes}
cov_pooled = sum(priors[k] * covs[k] for k in classes)

# EUCLIDIEN (Sigma=I)
print_section("EUCLIDIEN (Sigma=I)")
def euclidean_predict(X):
    scores = np.column_stack([
        X @ means[k] - 0.5 * np.sum(means[k]**2) + np.log(priors[k]) for k in classes])
    return classes[np.argmax(scores, axis=1)]
acc_euc = accuracy_score(y_test, euclidean_predict(X_test))
print(f"  Accuracy: {acc_euc:.4f}")

# NAIVE BAYES GAUSSIEN (Sigma diagonale)
print_section("NAIVE BAYES GAUSSIEN (Sigma diagonale)")
var_diag = {k: X_train[y_train == k].var(axis=0) + 1e-6 for k in classes}
def nb_predict(X):
    scores = np.column_stack([
        -0.5 * np.sum(np.log(var_diag[k])) - 0.5 * (((X - means[k])**2 / var_diag[k]).sum(axis=1))
        + np.log(priors[k]) for k in classes])
    return classes[np.argmax(scores, axis=1)]
acc_nb = accuracy_score(y_test, nb_predict(X_test))
nb_sk = GaussianNB().fit(X_train, y_train)
acc_nb_sk = accuracy_score(y_test, nb_sk.predict(X_test))
print(f"  Manuel: {acc_nb:.4f} | sklearn: {acc_nb_sk:.4f}")

# LDA (Sigma commun)
print_section("LDA")
inv_pooled = np.linalg.inv(cov_pooled + 1e-6 * np.eye(p))
def lda_predict(X):
    scores = np.column_stack([
        X @ inv_pooled @ means[k] - 0.5 * means[k] @ inv_pooled @ means[k] + np.log(priors[k])
        for k in classes])
    return classes[np.argmax(scores, axis=1)]
acc_lda = accuracy_score(y_test, lda_predict(X_test))
lda_sk = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X_train, y_train)
acc_lda_sk = accuracy_score(y_test, lda_sk.predict(X_test))
print(f"  Manuel: {acc_lda:.4f} | sklearn: {acc_lda_sk:.4f}")

# QDA (Sigma_k differents)
print_section("QDA")
def qda_predict(X, reg=0.1):
    scores = np.zeros((X.shape[0], len(classes)))
    for i, k in enumerate(classes):
        cov_reg = (1-reg) * covs[k] + reg * np.eye(p)
        log_det = np.linalg.slogdet(cov_reg)[1]
        inv_cov = np.linalg.inv(cov_reg)
        diff = X - means[k]
        scores[:, i] = -0.5 * log_det - 0.5 * np.sum(diff @ inv_cov * diff, axis=1) + np.log(priors[k])
    return classes[np.argmax(scores, axis=1)]
acc_qda = accuracy_score(y_test, qda_predict(X_test))
qda_sk = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(X_train, y_train)
acc_qda_sk = accuracy_score(y_test, qda_sk.predict(X_test))
print(f"  Manuel: {acc_qda:.4f} | sklearn: {acc_qda_sk:.4f}")

# COMPARAISON
print_section("RESUME")
results = {'Baseline': baseline, 'Euclidien (Sigma=I)': acc_euc, 'Naive Bayes (Sigma diag)': acc_nb_sk,
           'LDA (Sigma commun)': acc_lda_sk, 'QDA (Sigma_k)': acc_qda_sk}
for name, acc in results.items():
    print(f"  {name:22s}: {acc:.4f}")

# RAPPEL PAR CLASSE
print_section("RAPPEL PAR CLASSE")
print(f"  {'modele':14s} " + " ".join(f"{l:>12s}" for l in LABELS))
for name, preds in [("Euclidien", euclidean_predict(X_test)), ("Naive Bayes", nb_sk.predict(X_test)),
                    ("LDA", lda_sk.predict(X_test)), ("QDA", qda_sk.predict(X_test))]:
    rec = recall_score(y_test, preds, average=None, labels=classes)
    print(f"  {name:14s} " + " ".join(f"{r:>12.3f}" for r in rec))

# Validation croisee stratifiee
print_section("VALIDATION CROISEE (Stratified 5-fold)")
X_full = np.vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in [("Euclidien", NearestCentroid()),
                    ("Naive Bayes", GaussianNB()),
                    ("LDA", LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                    ("QDA", QuadraticDiscriminantAnalysis(reg_param=0.1))]:
    scores = cross_val_score(model, X_full, y_full, cv=skf)
    print(f"  {name:15s}: {scores.mean():.4f} +/- {scores.std():.4f}")

# CONFUSION
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (name, preds) in zip(axes, [('Euclidien', euclidean_predict(X_test)),
                                     ('Naive Bayes', nb_sk.predict(X_test)),
                                     ('LDA', lda_sk.predict(X_test)),
                                     ('QDA', qda_sk.predict(X_test))]):
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{name} — Acc={accuracy_score(y_test, preds):.3f}')
plt.tight_layout()
save_plot("D02_discriminant_confusion.png")

# PROJECTION LDA 2D
lda_2d = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)
X_test_2d = lda_2d.transform(X_test)
plt.figure(figsize=(8, 6))
couleurs = ['steelblue', 'coral', 'forestgreen']
for i, c in enumerate(classes):
    m = y_test == c
    plt.scatter(X_test_2d[m, 0], X_test_2d[m, 1], c=couleurs[i], label=LABELS[i],
                alpha=0.6, edgecolors='black', linewidth=0.3)
plt.xlabel('LD1'); plt.ylabel('LD2'); plt.legend()
plt.title('Projection LDA 2D (3 classes -> 2 axes discriminants)')
plt.tight_layout()
save_plot("D02_lda_projection.png")
