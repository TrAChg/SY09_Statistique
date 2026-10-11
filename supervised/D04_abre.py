import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import print_section, save_plot, BASE_DIR

print_section("D04 — ARBRE DE DECISION (CART - POST-ÉLAGAGE)")

# 1. Load Data
X_train = np.loadtxt(os.path.join(BASE_DIR, "X_train.csv"), delimiter=",")
X_test = np.loadtxt(os.path.join(BASE_DIR, "X_test.csv"), delimiter=",")
y_train = np.loadtxt(os.path.join(BASE_DIR, "y_train.csv"), delimiter=",").astype(int)
y_test = np.loadtxt(os.path.join(BASE_DIR, "y_test.csv"), delimiter=",").astype(int)
feature_names = pd.read_csv(os.path.join(BASE_DIR, "feature_names.csv")).iloc[:, 0].tolist()

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Baseline: {np.bincount(y_test).max()/len(y_test):.4f}")

# ==========================================
# 2. TÌM CHUỖI CÂY VÀ LAMBDA (ccp_alpha)
# ==========================================
print_section("POST-ÉLAGAGE — Tìm chuỗi lambda")

# Huấn luyện một cây khổng lồ (A0) để lấy danh sách các điểm cắt lambda
clf_base = DecisionTreeClassifier(random_state=42)
path = clf_base.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Bỏ giá trị alpha lớn nhất cuối cùng (vì nó sẽ tỉa sạch cây chỉ còn 1 nút gốc C0)
ccp_alphas = ccp_alphas[:-1]
print(f"  Số lượng ngưỡng lambda cần thử nghiệm: {len(ccp_alphas)}")

# ==========================================
# 3. TÌM LAMBDA TỐI ƯU BẰNG GRIDSEARCHCV
# ==========================================
print_section("XÁC THỰC CHÉO — Chọn cây tối ưu")

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"ccp_alpha": ccp_alphas},
    cv=StratifiedKFold(10, shuffle=True, random_state=42),
    return_train_score=True
)
grid.fit(X_train, y_train)

best_alpha = grid.best_params_["ccp_alpha"]
tree = grid.best_estimator_ # Đây chính là cây A* tối ưu
best_depth = tree.get_depth()

print(f"  Lambda (ccp_alpha) tối ưu: {best_alpha:.5f}")
print(f"  Độ sâu của cây sau khi tỉa: {best_depth}")
print(f"  Accuracy CV (Validation): {grid.best_score_:.4f}")

# Đánh giá trên tập Test
acc_test = accuracy_score(y_test, tree.predict(X_test))
print(f"  Estimation non biaisée (Test Accuracy): {acc_test:.4f}")

# Vẽ biểu đồ Accuracy vs Alpha
plt.figure(figsize=(8, 5))
cv_means = grid.cv_results_["mean_test_score"]
train_means = grid.cv_results_["mean_train_score"]
plt.plot(ccp_alphas, train_means, marker="o", color="steelblue", label="Train (CV)", drawstyle="steps-post")
plt.plot(ccp_alphas, cv_means, marker="o", color="coral", label="Validation (CV)", drawstyle="steps-post")
plt.axvline(best_alpha, color="green", ls="--", label=f"alpha={best_alpha:.4f}")
plt.xlabel("lambda (ccp_alpha)"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Arbre — Accuracy vs Lambda (Post-élagage)")
save_plot("D04_ccp_alpha.png")

# ==========================================
# 4. TRỰC QUAN HÓA & ĐÁNH GIÁ (Giữ nguyên logic của bạn)
# ==========================================
print_section("IMPORTANCES DES VARIABLES")
imp = pd.DataFrame({"feature": feature_names, "importance": tree.feature_importances_}
                   ).sort_values("importance", ascending=False)
print(imp.round(4).to_string(index=False))

plt.figure(figsize=(10, 5))
top = imp.head(10).iloc[::-1]
plt.barh(top["feature"], top["importance"], color="steelblue")
plt.xlabel("Importance"); plt.title("Importances — Arbre de décision optimisé")
plt.tight_layout()
save_plot("D04_importances.png")

# Vẽ cây tối ưu (hiển thị toàn bộ cây vì đã được tỉa gọn)
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_names, class_names=["non-buv", "weekend", "eleve"],
          filled=True, rounded=True, fontsize=10)
plt.title(f"Arbre de décision optimal (Post-élagage, alpha={best_alpha:.4f})")
save_plot("D04_arbre_optimal.png")

print_section("RAPPEL PAR CLASSE")
LABELS = ["non-buveurs", "weekend", "eleve"]
classes = np.unique(y_train)
rec = recall_score(y_test, tree.predict(X_test), average=None, labels=classes)
for l, r in zip(LABELS, rec):
    print(f"  rappel {l:12s}: {r:.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, tree.predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Arbre Optimal — Acc={acc_test:.3f}")
plt.tight_layout()
save_plot("D04_confusion.png")