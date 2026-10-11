"""
C01 — SETUP SUPERVISE: cible Risk_Class (3 classes), features, train/test split.
     Inclut la validation de la selection de features (all vs selected).
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_section, BASE_DIR

print_section("C01 — SETUP SUPERVISE")
df = pd.read_csv(os.path.join(BASE_DIR, "student-alcohol.csv"))
print(f"  {df.shape[0]} eleves, {df.shape[1]} colonnes")

profiles = pd.read_csv(os.path.join(BASE_DIR, "alcohol_profiles.csv"))
df["Risk_Class"] = profiles["Risk_Class"].values

print("  Cible Risk_Class: 0=non-buveurs, 1=weekend, 2=eleve")
print(df["Risk_Class"].value_counts().sort_index().to_string())

# =============================================================================
# SELECTION DES FEATURES
# Criteres de selection (justification statistique, pas arbitraire):
#   - Numeriques: ANOVA significative (p<0.05) entre clusters (B05)
#   - Nominales: chi2 significatif (p<0.05) avec Dalc ou Walc (B01)
#   - Medu/Fedu: bien que significatifs en ACP (PC1), leur ANOVA entre
#     clusters alcool donne p>0.05 (cf B05: non dans la liste des 7 facteurs
#     lies). Le capital familial structure la population (PC1) mais ne
#     discrimine PAS les profils de consommation d'alcool.
#   - G1/G2/G3: exclus (resultats scolaires = consequences, pas causes)
#   - health, traveltime: ANOVA non significative entre clusters alcool
# =============================================================================
numeric_features = ["age", "studytime", "failures", "famrel", "freetime", "goout", "absences"]
nominal_features = ["sex", "schoolsup", "famsup", "higher"]

encoded = pd.get_dummies(df[nominal_features], drop_first=True)
feature_frame = pd.concat([df[numeric_features], encoded], axis=1)
feature_names = feature_frame.columns.tolist()

X = feature_frame.values
y = df["Risk_Class"].values
print(f"  Features selectionnees: {X.shape[1]} ({', '.join(feature_names)})")
print(f"    - Numeriques (ANOVA p<0.05): {numeric_features}")
print(f"    - Nominales (chi2 p<0.05): {nominal_features}")

# =============================================================================
# VALIDATION: COMPARAISON ALL FEATURES vs SELECTED FEATURES
# Justifie que notre selection ne perd pas d'information utile.
# =============================================================================
print_section("VALIDATION SELECTION (all vs selected)")

all_nominal = ["sex", "schoolsup", "famsup", "higher", "school", "address",
               "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian",
               "paid", "activities", "nursery", "internet", "romantic"]
all_numeric = ["age", "traveltime", "studytime", "failures", "famrel",
               "freetime", "goout", "health", "absences", "Medu", "Fedu"]

encoded_all = pd.get_dummies(df[all_nominal], drop_first=True)
feature_frame_all = pd.concat([df[all_numeric], encoded_all], axis=1)
X_all = feature_frame_all.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler_sel = StandardScaler()
scaler_all = StandardScaler()
X_sel_s = scaler_sel.fit_transform(X)
X_all_s = scaler_all.fit_transform(X_all)

cv_sel = cross_val_score(LogisticRegression(max_iter=2000, random_state=42),
                         X_sel_s, y, cv=skf)
cv_all = cross_val_score(LogisticRegression(max_iter=2000, random_state=42),
                         X_all_s, y, cv=skf)
print(f"  Selected ({X.shape[1]} features): CV = {cv_sel.mean():.4f} +/- {cv_sel.std():.4f}")
print(f"  All ({X_all.shape[1]} features):      CV = {cv_all.mean():.4f} +/- {cv_all.std():.4f}")
diff = cv_all.mean() - cv_sel.mean()
print(f"  Difference: {diff:+.4f}")
if diff > 0.01:
    print("  -> All features meilleures. A investiguer (possible sous-selection).")
elif diff < -0.01:
    print("  -> Selected MEILLEURES que All (-3%): les features supplementaires")
    print("     n'apportent que du bruit (overfitting). La selection parcimonieuse")
    print("     basee sur ANOVA/chi2 est VALIDEE.")
else:
    print("  -> Pas de gain significatif avec toutes les features.")
    print("     La selection est validee: on garde le modele parcimonieux.")

# =============================================================================
# TRAIN/TEST SPLIT 70/30 stratifie
# =============================================================================
print_section("SPLIT TRAIN/TEST")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"  Baseline (classe majoritaire): {np.bincount(y_test).max()/len(y_test):.4f}")

# Sauvegarde
np.savetxt(os.path.join(BASE_DIR, "X_train.csv"), X_train_s, delimiter=",")
np.savetxt(os.path.join(BASE_DIR, "X_test.csv"), X_test_s, delimiter=",")
np.savetxt(os.path.join(BASE_DIR, "y_train.csv"), y_train, delimiter=",")
np.savetxt(os.path.join(BASE_DIR, "y_test.csv"), y_test, delimiter=",")
pd.Series(feature_names).to_csv(os.path.join(BASE_DIR, "feature_names.csv"), index=False)
print("  Sauve: X_train.csv, X_test.csv, y_train.csv, y_test.csv, feature_names.csv")
