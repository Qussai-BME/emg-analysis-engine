"""
minirocket.py v6.0 TURBO — MiniROCKET with sklearn RidgeClassifierCV
====================================================================
BLAS-optimized MiniROCKET + sklearn RidgeClassifierCV + StandardScaler.

v6.0: Same core as v5.3 (predict bug fixed).
  - StandardScaler on PPV features (critical for Ridge stability)
  - sklearn GCV is more robust than numpy eigenvalue approach
  - BLAS matmul for convolution (fast)
  - Per-window normalization handled in minirocket_loso.py

v5.3 fix: predict() was passing 3D X to clf instead of 2D X_feat.

If sklearn is NOT available, falls back to numpy RidgeClassifierCV.
"""

import numpy as np
import gc
import time
import sys
import traceback

# Try to import sklearn
try:
    from sklearn.linear_model import RidgeClassifierCV as SklearnRidgeCV
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ======================================================================
#  MiniRocket — Feature Transformer
# ======================================================================
class MiniRocket:
    """
    MiniROCKET: Random Convolutional Kernel Transform (numpy-only).

    - Deterministic weight patterns (matches reference)
    - BLAS matmul for fast convolution
    - One channel per kernel (MiniROCKET, not ROCKET)
    """

    VERSION = "7.0"

    def __init__(self, num_kernels=10000, kernel_length=9, proportion=0.5,
                 n_channels=None, random_state=42):
        self.num_kernels = num_kernels
        self.kernel_length = kernel_length
        self.proportion = proportion
        self.n_channels = n_channels
        self.random_state = random_state
        self.BIAS_SAMPLE_LIMIT = 500
        self.TRANSFORM_BATCH = 500

    def _generate_kernels(self, T):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        C = self.n_channels
        L = self.kernel_length

        max_power = int(np.floor(np.log2((T - 1) / max(L - 1, 1))))
        num_dilations = max(max_power, 1)
        self.unique_dilations = np.array([2 ** i for i in range(num_dilations)],
                                         dtype=np.int32)

        # Deterministic weight patterns
        _num_possible = 2 ** (int(np.ceil(np.log2(L + 1))) - 1)
        all_combos = np.array(
            [[int(w) for w in format(i, f'0{L}b')]
             for i in range(2 ** L)],
            dtype=np.float32
        )
        all_combos = 2.0 * all_combos - 1.0  # {-1, +1}
        weight_patterns = all_combos[:: 2**L // _num_possible]

        base_k = self.num_kernels // num_dilations
        remainder = self.num_kernels % num_dilations

        dilations_list = []
        weights_list = []
        channels_list = []
        kernel_idx = 0

        for d_idx, d in enumerate(self.unique_dilations):
            n_k = base_k + (1 if d_idx < remainder else 0)
            for _ in range(n_k):
                dilations_list.append(d)
                channels_list.append(np.random.randint(0, C))
                w = weight_patterns[kernel_idx % _num_possible].copy()
                weights_list.append(w)
                kernel_idx += 1

        self.dilations = np.array(dilations_list, dtype=np.int32)
        self.weights  = np.array(weights_list, dtype=np.float32)
        self.channels = np.array(channels_list, dtype=np.int32)
        self.biases   = np.zeros(self.num_kernels, dtype=np.float32)

        print(f"    [MiniRocket] Generated {self.num_kernels} kernels "
              f"({num_dilations} dilations, {_num_possible} weight patterns, "
              f"d=[{','.join(str(int(x)) for x in self.unique_dilations)}])")

    def _compute_biases(self, X):
        N, C, T = X.shape
        L = self.kernel_length

        N_sub = min(N, self.BIAS_SAMPLE_LIMIT)
        if N_sub < N:
            rng = np.random.RandomState(self.random_state + 1 if self.random_state else 0)
            idx = rng.choice(N, N_sub, replace=False)
        else:
            idx = np.arange(N)
        X_sub = np.ascontiguousarray(X[idx], dtype=np.float32)

        t0 = time.time()
        n_groups = len(self.unique_dilations)

        for g_idx, d in enumerate(self.unique_dilations):
            dt0 = time.time()
            d_int = int(d)

            mask = (self.dilations == d)
            global_idx = np.where(mask)[0]
            K_d = len(global_idx)
            T_out = T - (L - 1) * d_int

            if T_out <= 0:
                continue

            W_d = self.weights[mask]
            ch_d = self.channels[mask]

            for c in range(C):
                c_mask = (ch_d == c)
                K_c = int(c_mask.sum())
                if K_c == 0:
                    continue

                W_c = W_d[c_mask]
                gidx_c = global_idx[c_mask]
                X_c = X_sub[:, c, :]

                lag_flat = np.empty((L, N_sub * T_out), dtype=np.float32)
                for j in range(L):
                    lag_flat[j] = X_c[:, j * d_int : j * d_int + T_out].ravel()

                conv = W_c @ lag_flat
                quantiles = np.quantile(conv, self.proportion, axis=1)
                self.biases[gidx_c] = -quantiles.astype(np.float32)

                del lag_flat, conv

            dt = time.time() - dt0
            eta = dt * (n_groups - g_idx - 1)
            print(f"    [Bias] {g_idx+1}/{n_groups}: d={d}, {K_d} kernels, "
                  f"{dt:.1f}s (ETA {eta:.0f}s)")
            gc.collect()

        print(f"    [MiniRocket] All biases computed ({time.time()-t0:.1f}s)")

    def _batched_transform(self, X):
        N, C, T = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)
        K = self.num_kernels
        L = self.kernel_length
        ppv = np.zeros((N, K), dtype=np.float32)

        t0 = time.time()
        n_groups = len(self.unique_dilations)
        batch_size = self.TRANSFORM_BATCH

        for g_idx, d in enumerate(self.unique_dilations):
            dt0 = time.time()
            d_int = int(d)

            mask = (self.dilations == d)
            global_idx = np.where(mask)[0]
            K_d = len(global_idx)
            T_out = T - (L - 1) * d_int

            if T_out <= 0:
                continue

            W_d = self.weights[mask]
            b_d = self.biases[mask]
            ch_d = self.channels[mask]

            for c in range(C):
                c_mask = (ch_d == c)
                K_c = int(c_mask.sum())
                if K_c == 0:
                    continue

                W_c = W_d[c_mask]
                b_c = b_d[c_mask]
                gidx_c = global_idx[c_mask]
                X_c = X[:, c, :]

                sb = 0
                while sb < N:
                    eb = min(sb + batch_size, N)
                    B = eb - sb
                    X_sub = X_c[sb:eb]

                    try:
                        lag_flat = np.empty((L, B * T_out), dtype=np.float32)
                        for j in range(L):
                            lag_flat[j] = X_sub[:, j * d_int : j * d_int + T_out].ravel()

                        conv = W_c @ lag_flat
                        conv += b_c[:, None]

                        ppv_sub = np.mean(
                            (conv > 0).reshape(K_c, B, T_out), axis=2
                        )
                        ppv[sb:eb, gidx_c] = ppv_sub.T

                        del lag_flat, conv

                    except MemoryError:
                        if batch_size <= 32:
                            print(f"\n    [ERROR] MemoryError at batch=32")
                            traceback.print_exc()
                            break
                        old_bs = batch_size
                        batch_size = max(32, batch_size // 2)
                        print(f"\n    [Memory] batch {old_bs}->{batch_size}")
                        gc.collect()
                        continue

                    sb = eb

            dt = time.time() - dt0
            elapsed = time.time() - t0
            eta = (elapsed / (g_idx + 1)) * (n_groups - g_idx - 1) if g_idx > 0 else dt * (n_groups - 1)
            print(f"    [Transform] {g_idx+1}/{n_groups}: d={d}, {K_d} kernels "
                  f"({dt:.1f}s, total ETA {eta:.0f}s)")
            gc.collect()

        ppv = np.nan_to_num(ppv, nan=0.0, posinf=1.0, neginf=0.0)
        return ppv

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        N, C, T = X.shape
        if self.n_channels is None:
            self.n_channels = C
        elif self.n_channels != C:
            print(f"    [Warning] n_channels: expected {self.n_channels}, got {C}")
            self.n_channels = C
        self._generate_kernels(T)
        self._compute_biases(X)
        return self

    def transform(self, X):
        return self._batched_transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# ======================================================================
#  RidgeClassifierCV — sklearn (preferred) or numpy fallback
# ======================================================================
class RidgeClassifierCV:
    """
    Ridge Classifier with GCV alpha selection.

    v5.2: Uses sklearn RidgeClassifierCV if available (well-tested).
    Falls back to numpy implementation if sklearn not installed.
    """

    VERSION = "3.0"

    def __init__(self, alphas=None, random_state=42):
        if HAS_SKLEARN:
            self.alphas = alphas or np.logspace(-3, 6, 25).tolist()
            self._sklearn = SklearnRidgeCV(
                alphas=self.alphas,
                scoring='accuracy',
                cv=None,  # GCV (Leave-One-Out)
            )
            self._backend = 'sklearn'
        else:
            self.alphas = alphas or np.logspace(-3, 6, 20).tolist()
            self._sklearn = None
            self._backend = 'numpy'
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if self._backend == 'sklearn':
            self._sklearn.fit(X, y)
            self.classes_ = self._sklearn.classes_
            self.coef_ = self._sklearn.coef_
            self.intercept_ = self._sklearn.intercept_
            self.alpha_ = self._sklearn.alpha_
            train_acc = self._sklearn.score(X, y)
            print(f"    [RidgeClassifierCV] SKLEARN alpha={self.alpha_:.4f}, "
                  f"{len(self.classes_)} classes, train_acc={train_acc:.4f}")
        else:
            self._fit_numpy(X, y)

        return self

    def _fit_numpy(self, X, y):
        """Numpy fallback — GCV via eigenvalue decomposition."""
        N, P = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        Y = np.zeros((N, K), dtype=np.float64)
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0

        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        Xc = X - X_mean
        Yc = Y - Y_mean

        XTX = Xc.T @ Xc
        XTY = Xc.T @ Yc

        best_alpha = self.alphas[0]
        best_score = np.inf

        if P <= 5000:
            eigvals, V = np.linalg.eigh(XTX)
            eigvals = np.maximum(eigvals, 0)
            VTY = V.T @ XTY
            for alpha in self.alphas:
                D_inv = 1.0 / (eigvals + alpha)
                coef = V @ (D_inv[:, None] * VTY)
                residuals = Yc - Xc @ coef
                RSS = np.sum(residuals ** 2)
                df = np.sum(eigvals / (eigvals + alpha))
                gcv = RSS * N / max(N - df, 1) ** 2
                if np.isfinite(gcv) and gcv < best_score:
                    best_score = gcv
                    best_alpha = alpha
            del eigvals, V, VTY
        else:
            eye_P = np.eye(P, dtype=np.float64)
            for alpha in self.alphas:
                try:
                    coef = np.linalg.solve(XTX + alpha * eye_P, XTY)
                    RSS = float(np.sum((Yc - Xc @ coef) ** 2))
                    if RSS < best_score:
                        best_score = RSS
                        best_alpha = alpha
                except Exception:
                    continue

        eye_P = np.eye(P, dtype=np.float64)
        self.coef_ = np.linalg.solve(XTX + best_alpha * eye_P, XTY)
        self.intercept_ = Y_mean - X_mean @ self.coef_
        self.alpha_ = best_alpha

        pred = np.argmax(Xc @ self.coef_, axis=1)
        true = np.argmax(Yc, axis=1)
        train_acc = float(np.mean(pred == true))
        print(f"    [RidgeClassifierCV] NUMPY alpha={best_alpha:.4f}, "
              f"{K} classes, train_acc={train_acc:.4f}")

        del XTX, XTY, Xc, Yc
        gc.collect()

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._backend == 'sklearn':
            return self._sklearn.decision_function(X)
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        if self._backend == 'sklearn':
            return self._sklearn.predict(X)
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


# ======================================================================
#  MiniRocketPipeline
# ======================================================================
class MiniRocketPipeline:
    """
    Pipeline: MiniRocket (BLAS) -> StandardScaler -> RidgeClassifierCV.

    v7.0: Same as v6.0 + CORAL domain adaptation handled in minirocket_loso.py.
    """

    VERSION = "7.0 TURBO"

    def __init__(self, num_kernels=10000, kernel_length=9, proportion=0.5,
                 alphas=None, random_state=42):
        self.rocket = MiniRocket(
            num_kernels=num_kernels,
            kernel_length=kernel_length,
            proportion=proportion,
            random_state=random_state,
        )

        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            print("    [Pipeline] Using sklearn StandardScaler + RidgeClassifierCV", flush=True)
        else:
            self.scaler = None
            print("    [Pipeline] Using numpy-only RidgeClassifierCV (no sklearn)", flush=True)

        self.clf = RidgeClassifierCV(alphas=alphas, random_state=random_state)

    def fit(self, X, y):
        X_feat = self.rocket.fit_transform(X, y)

        # Diagnostics on raw PPV
        print(f"    [Diag] Raw PPV: mean={X_feat.mean():.4f}, std={X_feat.std():.4f}, "
              f"min={X_feat.min():.4f}, max={X_feat.max():.4f}", flush=True)

        if self.scaler is not None:
            X_feat = self.scaler.fit_transform(X_feat)
            print(f"    [Diag] Scaled PPV: mean={X_feat.mean():.4f}, std={X_feat.std():.4f}",
                  flush=True)

        self.clf.fit(X_feat, y)
        return self

    def predict(self, X):
        X_feat = self.rocket.transform(X)
        if self.scaler is not None:
            X_feat = self.scaler.transform(X_feat)
        return self.clf.predict(X_feat)

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

    def decision_function(self, X):
        X_feat = self.rocket.transform(X)
        if self.scaler is not None:
            X_feat = self.scaler.transform(X_feat)
        return self.clf.decision_function(X_feat)

    @property
    def classes_(self):
        return self.clf.classes_
