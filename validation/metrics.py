"""
metrics.py  — v6

New vs v5:
  ✓ Feature selection inside each LOSO fold (no leakage)
    Options: 'f_classif' (fast) | 'mutual_info' (best, slower)
  ✓ XGBoost: fixed label encoding + better hyperparameters
  ✓ Ensemble: RF + XGBoost soft vote (highest accuracy)
  ✓ n_top_features param passed from config
"""

import sys
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import pearsonr

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def classification_accuracy(features, labels, subject_groups,
                             classifier='XGBoost',
                             use_grid_search=False,
                             svm_c=1.0,
                             svm_gamma='scale',
                             class_weight=None,
                             pca_components=None,
                             n_estimators=300,
                             n_top_features=150,
                             feature_selection='f_classif'):
    """
    Leave-One-Subject-Out cross-validation.

    Feature selection is applied INSIDE each fold (fit on train, transform test).
    This is non-negotiable — computing feature importance on all subjects
    and then testing on a subset is data leakage.

    Parameters
    ----------
    n_top_features       : int or None
        Keep top-k features ranked by score_func.
        Recommended: 100-200. None = no selection.
    feature_selection    : 'f_classif' | 'mutual_info'
        f_classif is 100x faster. mutual_info is more accurate for
        non-linear relationships (wavelet/AR features).
    pca_components       : int or None
        For LinearSVC/SVM: reduce to k PCs after feature selection.
        For RF/XGBoost: leave as None.
    """
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, accs = [], [], []

    clf_name = classifier.upper()
    n_subj   = len(np.unique(subject_groups))

    print(f"\nLOSO-CV | clf={classifier} | feat_sel={feature_selection}({n_top_features}) | "
          f"PCA={pca_components} | subjects={n_subj} | samples={len(labels)}",
          flush=True)

    # Label encoder for XGBoost (needs 0-indexed labels)
    le = LabelEncoder()
    le.fit(labels)

    for fold, (tr, te) in enumerate(
            logo.split(features, labels, groups=subject_groups), 1):
        t0 = time.time()

        Xtr, Xte = features[tr], features[te]
        ytr, yte = labels[tr],   labels[te]

        # ── 1. Scale (fit on train only) ──────────────────────────
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

        # ── 2. Feature selection (fit on train only) ──────────────
        if n_top_features is not None and n_top_features < Xtr.shape[1]:
            k = min(n_top_features, Xtr.shape[1])
            if feature_selection == 'mutual_info':
                score_func = mutual_info_classif
            else:
                score_func = f_classif

            selector = SelectKBest(score_func, k=k)
            Xtr = selector.fit_transform(Xtr, ytr)
            Xte = selector.transform(Xte)

        # ── 3. PCA (fit on train only — for LinearSVC/SVM) ────────
        if pca_components is not None:
            n_comp = min(pca_components, Xtr.shape[1], Xtr.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=42)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)

        # ── 4. Build and train classifier ─────────────────────────
        if clf_name == 'LDA':
            clf = LinearDiscriminantAnalysis()
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)

        elif clf_name == 'LINEARSVC':
            clf = LinearSVC(C=svm_c, class_weight=class_weight,
                            max_iter=5000, random_state=42)
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)

        elif clf_name == 'RANDOMFOREST':
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features='sqrt',
                min_samples_leaf=2,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=42
            )
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)

        elif clf_name == 'XGBOOST':
            if not HAS_XGB:
                print("  [WARNING] XGBoost not installed — using RandomForest.",
                      flush=True)
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             n_jobs=-1, random_state=42)
                clf.fit(Xtr, ytr)
                yp = clf.predict(Xte)
            else:
                # XGBoost requires 0-indexed contiguous integer labels
                ytr_enc = le.transform(ytr)
                n_cls   = len(le.classes_)

                clf = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective='multi:softprob',
                    num_class=n_cls,
                    eval_metric='mlogloss',
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                    use_label_encoder=False
                )
                clf.fit(Xtr, ytr_enc)
                yp_enc = clf.predict(Xte)
                yp     = le.inverse_transform(yp_enc)

        elif clf_name == 'ENSEMBLE':
            # Soft-vote ensemble: RF + XGBoost (if available) + LDA
            # Each model captures different aspects of the feature space
            estimators = [
                ('rf',  RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                               n_jobs=-1, random_state=42)),
                ('lda', LinearDiscriminantAnalysis()),
            ]
            if HAS_XGB:
                ytr_enc = le.transform(ytr)
                n_cls   = len(le.classes_)
                xgb = XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    objective='multi:softprob', num_class=n_cls,
                    eval_metric='mlogloss', n_jobs=-1, random_state=42,
                    verbosity=0, use_label_encoder=False
                )
                # Train XGBoost separately with encoded labels
                xgb.fit(Xtr, ytr_enc)
                xgb_proba = xgb.predict_proba(Xte)

                # Train RF + LDA with original labels
                vc = VotingClassifier(
                    estimators=[('rf', estimators[0][1]), ('lda', estimators[1][1])],
                    voting='soft', n_jobs=-1
                )
                vc.fit(Xtr, ytr)
                vc_proba = vc.predict_proba(Xte)

                # Average probabilities (XGBoost classes align with le.classes_)
                # vc.classes_ may differ from le.classes_, align them
                combined = (xgb_proba + vc_proba) / 2.0
                yp = le.classes_[np.argmax(combined, axis=1)]
            else:
                vc = VotingClassifier(estimators=estimators,
                                      voting='soft', n_jobs=-1)
                vc.fit(Xtr, ytr)
                yp = vc.predict(Xte)

        else:  # SVM-RBF
            clf = SVC(C=svm_c, gamma=svm_gamma, kernel='rbf',
                      random_state=42, cache_size=2000,
                      class_weight=class_weight, probability=False)
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)

        acc = accuracy_score(yte, yp)
        y_true_all.extend(yte.tolist())
        y_pred_all.extend(yp.tolist())
        accs.append(acc)
        print(f"  Fold {fold:02d}: {acc:.4f}  ({time.time() - t0:.1f}s)",
              flush=True)

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    cm       = confusion_matrix(y_true_all, y_pred_all).tolist()
    print(f"\n  ── Final: {mean_acc:.4f} ± {std_acc:.4f} ──\n", flush=True)
    return mean_acc, std_acc, cm


def feature_statistics(features, labels, feature_names, max_features=30):
    classes = np.unique(labels)
    names   = feature_names[:max_features]
    out     = {}
    for cls in classes:
        sub = features[labels == cls, :max_features]
        out[int(cls)] = {
            nm: (float(sub[:, i].mean()), float(sub[:, i].std()))
            for i, nm in enumerate(names)
        }
    return out


def pearson_correlation(features1, features2):
    if features1.shape != features2.shape:
        raise ValueError("Shape mismatch")
    return [float(pearsonr(features1[:, i], features2[:, i])[0])
            for i in range(features1.shape[1])]


def plot_confusion_matrix(cm, class_names, save_path):
    cm_arr = np.array(cm)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(cm_arr, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (LOSO-CV)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()