import numpy as np


class GaussianWindowDiscriminant:
    """LDA/QDA window classifier using the dissertation discriminant score.

    Parameters
    ----------
    mode:
        "LDA" uses a pooled covariance matrix; "QDA" uses class-specific
        covariance matrices.

    covariance_reg:
        Optional diagonal loading:

            S_reg = S + covariance_reg * trace(S) / n_features * I.

        This is technical ridge stabilization.

    shrinkage_method:
        Covariance shrinkage target:
            "none"      - no shrinkage;
            "diagonal"  - shrinkage to diag(diag(S));
            "spherical" - shrinkage to mu * I, where mu = trace(S) / n_features.

    shrinkage_alpha:
        Shrinkage intensity:
            0.0 -> original covariance;
            1.0 -> only target covariance.

    Notes
    -----
    Baseline dissertation-style mode:
        covariance_reg=0.0,
        shrinkage_method="none",
        shrinkage_alpha=0.0

    Shrinkage and ridge regularization are technical modifications.
    """

    def __init__(
        self,
        mode: str = "LDA",
        covariance_reg: float = 0.0,
        shrinkage_method: str = "none",
        shrinkage_alpha: float = 0.0,
    ):
        if mode not in {"LDA", "QDA"}:
            raise ValueError("mode must be LDA or QDA")

        if covariance_reg < 0:
            raise ValueError("covariance_reg must be non-negative")

        if shrinkage_method not in {"none", "diagonal", "spherical"}:
            raise ValueError(
                "shrinkage_method must be one of: "
                "'none', 'diagonal', 'spherical'"
            )

        if not (0.0 <= shrinkage_alpha <= 1.0):
            raise ValueError("shrinkage_alpha must be in [0, 1]")

        self.mode = mode
        self.covariance_reg = float(covariance_reg)
        self.shrinkage_method = shrinkage_method
        self.shrinkage_alpha = float(shrinkage_alpha)

        self.classes_ = None
        self.means_ = {}
        self.covariances_ = {}
        self.inv_covariances_ = {}
        self.logdets_ = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        self.classes_ = np.unique(y)

        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required")

        n_samples, n_features = X.shape

        for cls in self.classes_:
            Xj = X[y == cls]

            if Xj.shape[0] <= 1:
                raise ValueError(f"Too few samples in class {cls}")

            self.means_[cls] = Xj.mean(axis=0)

        if self.mode == "QDA":
            for cls in self.classes_:
                Xj = X[y == cls]
                centered = Xj - self.means_[cls]

                S = centered.T @ centered / (Xj.shape[0] - 1)

                self._store_covariance(cls, S)

        else:
            pooled = np.zeros((n_features, n_features), dtype=float)

            for cls in self.classes_:
                Xj = X[y == cls]
                centered = Xj - self.means_[cls]
                pooled += centered.T @ centered

            denom = n_samples - len(self.classes_)

            if denom <= 0:
                raise ValueError("Not enough samples for pooled covariance")

            S = pooled / denom

            for cls in self.classes_:
                self._store_covariance(cls, S)

        return self

    def _apply_shrinkage(self, S):
        S = np.asarray(S, dtype=float)

        if self.shrinkage_method == "none" or self.shrinkage_alpha == 0.0:
            return S.copy()

        alpha = self.shrinkage_alpha
        n_features = S.shape[0]

        if self.shrinkage_method == "diagonal":
            target = np.diag(np.diag(S))

        elif self.shrinkage_method == "spherical":
            mu = np.trace(S) / n_features

            if not np.isfinite(mu) or mu <= 0:
                mu = 1.0

            target = mu * np.eye(n_features)

        else:
            raise ValueError(f"Unknown shrinkage_method: {self.shrinkage_method}")

        return (1.0 - alpha) * S + alpha * target

    def _apply_ridge(self, S):
        S = np.asarray(S, dtype=float)

        if self.covariance_reg <= 0:
            return S

        scale = np.trace(S) / S.shape[0]

        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        return S + self.covariance_reg * scale * np.eye(S.shape[0])

    def _store_covariance(self, cls, S):
        S = np.asarray(S, dtype=float)

        # 1. structured shrinkage
        S = self._apply_shrinkage(S)

        # 2. optional technical ridge for numerical stability
        S = self._apply_ridge(S)

        sign, logdet = np.linalg.slogdet(S)

        if sign <= 0:
            raise np.linalg.LinAlgError(
                f"Covariance matrix for class {cls} is not positive definite: "
                f"sign={sign}, logdet={logdet}. "
                f"mode={self.mode}, "
                f"covariance_reg={self.covariance_reg}, "
                f"shrinkage_method={self.shrinkage_method}, "
                f"shrinkage_alpha={self.shrinkage_alpha}"
            )

        self.covariances_[cls] = S
        self.inv_covariances_[cls] = np.linalg.inv(S)
        self.logdets_[cls] = logdet

    def window_scores(self, window):
        window = np.asarray(window, dtype=float)
        l = window.shape[0]

        scores = {}

        for cls in self.classes_:
            m = self.means_[cls]
            invS = self.inv_covariances_[cls]
            logdet = self.logdets_[cls]

            centered = window - m
            mahal = np.einsum("ij,jk,ik->i", centered, invS, centered)

            scores[cls] = l * logdet + mahal.sum()

        return scores

    def predict_window(self, window):
        scores = self.window_scores(window)
        return min(scores, key=scores.get)

    def predict_windows(self, windows):
        return np.asarray(
            [self.predict_window(w) for w in windows],
            dtype=int,
        )
        

def make_catboost_model(random_seed: int = 42, *, conservative: bool = True):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install catboost first: pip install catboost") from exc

    if conservative:
        return CatBoostClassifier(
            iterations=200,
            depth=2,
            learning_rate=0.03,
            l2_leaf_reg=30,
            loss_function="Logloss",
            random_seed=random_seed,
            verbose=0,
        )

    return CatBoostClassifier(
        iterations=200,
        depth=3,
        learning_rate=0.05,
        l2_leaf_reg=10,
        loss_function="Logloss",
        random_seed=random_seed,
        verbose=0,
    )


def make_catboost_tiny_model(random_seed: int = 42):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install catboost first: pip install catboost") from exc

    return CatBoostClassifier(
        iterations=30,
        depth=1,
        learning_rate=0.03,
        l2_leaf_reg=100,
        min_data_in_leaf=10,
        loss_function="Logloss",
        random_seed=random_seed,
        verbose=0,
    )


def make_svm_model(kind: str = "linear", random_seed: int = 42):
    """
    SVM baseline for window-level statistical features.

    kind:
        "linear" - linear SVM;
        "rbf"    - RBF SVM.
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ImportError("Install scikit-learn first: pip install scikit-learn") from exc

    if kind == "linear":
        svm = SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            random_state=random_seed,
        )
    elif kind == "rbf":
        svm = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=random_seed,
        )
    else:
        raise ValueError(f"Unknown SVM kind: {kind}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", svm),
        ]
    )


def make_classifier(name: str):
    if name == "LDA":
        return GaussianWindowDiscriminant(mode="LDA")

    if name == "QDA":
        return GaussianWindowDiscriminant(mode="QDA")

    if name.startswith("CatBoostTiny"):
        return make_catboost_tiny_model(random_seed=42)

    if name.startswith("CatBoostConservative"):
        return make_catboost_model(random_seed=42, conservative=True)

    if name.startswith("CatBoost"):
        return make_catboost_model(random_seed=42, conservative=False)

    if name.startswith("SVM_linear"):
        return make_svm_model(kind="linear", random_seed=42)

    if name.startswith("SVM_rbf"):
        return make_svm_model(kind="rbf", random_seed=42)

    raise ValueError(f"Unknown classifier: {name}")
