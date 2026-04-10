"""
metrics.py — v7.0

Improvements:
  + Added evaluate_model() supporting 'loso' and 'within_subject' strategies.
  + XGBoost now uses tree_method='hist' for speed.
  + Returns both accuracy and confusion matrix.
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
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
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
    Legacy LOSO wrapper (kept for backward compatibility).
    """
    return evaluate_model(
        features, labels, subject_groups,
        strategy='loso',
        classifier=classifier,
        svm_c=svm_c,
        svm_gamma=svm_gamma,
        class_weight=class_weight,
        pca_components=pca_components,
        n_estimators=n_estimators,
        n_top_features=n_top_features,
        feature_selection=feature_selection
    )


def evaluate_model(features, labels, subject_groups,
                   strategy='loso',
                   train_ratio=0.7,
                   random_state=42,
                   classifier='XGBoost',
                   svm_c=1.0,
                   svm_gamma='scale',
                   class_weight=None,
                   pca_components=None,
                   n_estimators=300,
                   n_top_features=150,
                   feature_selection='f_classif'):
    """
    Unified evaluation with support for both LOSO and within-subject splits.

    Parameters
    ----------
    strategy : 'loso' or 'within_subject'
        - 'loso': Leave-One-Subject-Out (subject_groups used as fold ids)
        - 'within_subject': For each unique subject, split windows randomly
          (train_ratio for training). Results are averaged over subjects.

    Returns
    -------
    mean_acc : float
    std_acc  : float
    cm       : list of list (confusion matrix)
    """
    if strategy not in ('loso', 'within_subject'):
        raise ValueError(f"Unknown strategy: {strategy}")

    unique_subjects = np.unique(subject_groups)
    n_subj = len(unique_subjects)
    clf_name = classifier.upper()

    print(f"\nEvaluation | strategy={strategy} | clf={classifier} | "
          f"feat_sel={feature_selection}({n_top_features}) | PCA={pca_components}",
          flush=True)

    # Label encoder for XGBoost (needs 0-indexed labels)
    le = LabelEncoder()
    le.fit(labels)

    y_true_all, y_pred_all, accs = [], [], []

    if strategy == 'loso':
        logo = LeaveOneGroupOut()
        splits = list(logo.split(features, labels, groups=subject_groups))
        print(f"LOSO folds: {len(splits)}", flush=True)
        for fold, (tr_idx, te_idx) in enumerate(splits, 1):
            t0 = time.time()
            Xtr, Xte = features[tr_idx], features[te_idx]
            ytr, yte = labels[tr_idx],   labels[te_idx]

            acc, yp = _train_and_evaluate(
                Xtr, ytr, Xte, yte, le,
                clf_name, svm_c, svm_gamma, class_weight,
                pca_components, n_estimators, n_top_features, feature_selection
            )
            accs.append(acc)
            y_true_all.extend(yte.tolist())
            y_pred_all.extend(yp.tolist())
            print(f"  Fold {fold:02d}: {acc:.4f}  ({time.time() - t0:.1f}s)",
                  flush=True)

    else:  # within_subject
        print(f"Within-subject splits: train_ratio={train_ratio}", flush=True)
        for subj_id in unique_subjects:
            subj_mask = subject_groups == subj_id
            X_subj = features[subj_mask]
            y_subj = labels[subj_mask]

            if len(np.unique(y_subj)) < 2:
                print(f"  Subject {subj_id}: insufficient classes, skipping.")
                continue

            # Split randomly
            Xtr, Xte, ytr, yte = train_test_split(
                X_subj, y_subj, train_size=train_ratio,
                random_state=random_state, stratify=y_subj
            )

            t0 = time.time()
            acc, yp = _train_and_evaluate(
                Xtr, ytr, Xte, yte, le,
                clf_name, svm_c, svm_gamma, class_weight,
                pca_components, n_estimators, n_top_features, feature_selection
            )
            accs.append(acc)
            y_true_all.extend(yte.tolist())
            y_pred_all.extend(yp.tolist())
            print(f"  Subject {subj_id}: {acc:.4f}  ({time.time() - t0:.1f}s)",
                  flush=True)

    if not accs:
        return 0.0, 0.0, []

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    cm       = confusion_matrix(y_true_all, y_pred_all).tolist()
    print(f"\n  ── Final: {mean_acc:.4f} ± {std_acc:.4f} ──\n", flush=True)
    return mean_acc, std_acc, cm


def _train_and_evaluate(Xtr, ytr, Xte, yte, le,
                        clf_name, svm_c, svm_gamma, class_weight,
                        pca_components, n_estimators, n_top_features, feature_selection):
    """Helper to avoid code duplication."""
    # Scale
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    # Feature selection
    if n_top_features is not None and n_top_features < Xtr.shape[1]:
        k = min(n_top_features, Xtr.shape[1])
        if feature_selection == 'mutual_info':
            score_func = mutual_info_classif
        else:
            score_func = f_classif
        selector = SelectKBest(score_func, k=k)
        Xtr = selector.fit_transform(Xtr, ytr)
        Xte = selector.transform(Xte)

    # PCA (optional)
    if pca_components is not None:
        n_comp = min(pca_components, Xtr.shape[1], Xtr.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)

    # Classifier
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
            n_estimators=n_estimators, max_features='sqrt',
            min_samples_leaf=2, class_weight=class_weight,
            n_jobs=-1, random_state=42
        )
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)

    elif clf_name == 'XGBOOST':
        if not HAS_XGB:
            clf = RandomForestClassifier(n_estimators=n_estimators,
                                         n_jobs=-1, random_state=42)
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
        else:
            ytr_enc = le.transform(ytr)
            n_cls = len(le.classes_)
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
                tree_method='hist',        # ← speedup
                n_jobs=-1,
                random_state=42,
                verbosity=0,
                use_label_encoder=False
            )
            clf.fit(Xtr, ytr_enc)
            yp_enc = clf.predict(Xte)
            yp = le.inverse_transform(yp_enc)

    elif clf_name == 'ENSEMBLE':
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                          n_jobs=-1, random_state=42)),
            ('lda', LinearDiscriminantAnalysis()),
        ]
        if HAS_XGB:
            ytr_enc = le.transform(ytr)
            n_cls = len(le.classes_)
            xgb = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective='multi:softprob', num_class=n_cls,
                eval_metric='mlogloss', tree_method='hist',
                n_jobs=-1, random_state=42, verbosity=0,
                use_label_encoder=False
            )
            xgb.fit(Xtr, ytr_enc)
            xgb_proba = xgb.predict_proba(Xte)

            vc = VotingClassifier(
                estimators=[('rf', estimators[0][1]), ('lda', estimators[1][1])],
                voting='soft', n_jobs=-1
            )
            vc.fit(Xtr, ytr)
            vc_proba = vc.predict_proba(Xte)

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
    return acc, yp


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
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()