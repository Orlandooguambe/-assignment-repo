

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# =========================================================
# Core: K-means from scratch
# =========================================================

class ScratchKMeans:
    """
    K-means scratch implementation

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K)
    n_init : int
        How many random initializations to try. Best (lowest SSE) kept.
    max_iter : int
        Max iterations per run
    tol : float
        Convergence tolerance on total centroid shift
    verbose : bool
        If True, prints progress

    Attributes (after fit)
    ----------
    cluster_centers_ : (K, d) final centroids
    labels_ : (n,) labels for training set (from the best run)
    inertia_ : float, SSE of the best run
    n_iter_ : int, iterations taken in the best run
    """

    def __init__(self, n_clusters=3, n_init=10, max_iter=300, tol=1e-4, verbose=False):
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = bool(self.verbose) if hasattr(self, "verbose") else bool(verbose)
        self.verbose = bool(verbose)

        # to be set after fit
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    # ---------- helpers ----------
    @staticmethod
    def _euclidean_sq(A, B):
        """
        Pairwise squared Euclidean distances between rows of A (n,d) and B (m,d).
        Returns (n, m) matrix.
        """
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        A2 = np.sum(A * A, axis=1, keepdims=True)  # (n,1)
        B2 = np.sum(B * B, axis=1, keepdims=True).T  # (1,m)
        return A2 + B2 - 2 * (A @ B.T)

    def _init_centroids_random(self, X, rng):
        """[Problem 1] Pick K random, unique data points as initial centroids."""
        n = X.shape[0]
        idx = rng.choice(n, size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _assign_clusters(self, X, centroids):
        """
        [Problem 3] Assign each point to the nearest centroid.
        Returns labels (n,)
        """
        d2 = self._euclidean_sq(X, centroids)  # (n, K)
        return np.argmin(d2, axis=1)

    def _update_centroids(self, X, labels):
        """
        [Problem 4] Move each centroid to the mean of its assigned points.
        If a cluster gets no points, re-initialize it to a random data point.
        """
        K = self.n_clusters
        d = X.shape[1]
        newC = np.zeros((K, d), dtype=float)
        for k in range(K):
            mask = (labels == k)
            if not np.any(mask):
                # empty cluster: fallback to a random point
                newC[k] = X[np.random.randint(0, X.shape[0])]
            else:
                newC[k] = X[mask].mean(axis=0)
        return newC

    @staticmethod
    def _sse(X, labels, centroids):
        """
        [Problem 2] Sum of Squared Errors (SSE / inertia).
        SSE = sum_k sum_{x in cluster k} ||x - mu_k||^2
        """
        sse = 0.0
        for k in range(centroids.shape[0]):
            diff = X[labels == k] - centroids[k]
            if diff.size:
                sse += np.sum(diff * diff)
        return float(sse)

    # ---------- training ----------
    def fit(self, X):
        """
        [Problem 5 & 6] Run K-means with n_init restarts and keep the best (lowest SSE).
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        best_sse = np.inf
        best = None

        rng_master = np.random.default_rng()

        for run in range(self.n_init):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            centroids = self._init_centroids_random(X, rng)
            prev_centroids = centroids.copy()
            converged = False

            for it in range(1, self.max_iter + 1):
                labels = self._assign_clusters(X, centroids)
                centroids = self._update_centroids(X, labels)

                shift = float(np.sum(np.linalg.norm(centroids - prev_centroids, axis=1)))
                if self.verbose and (it % max(1, self.max_iter // 5) == 0 or it == 1):
                    print(f"[run {run+1}/{self.n_init}] iter {it:4d} | total shift={shift:.6f}")

                if shift <= self.tol:
                    converged = True
                    break
                prev_centroids = centroids.copy()

            sse = self._sse(X, labels, centroids)
            if self.verbose:
                print(f"[run {run+1}] finished in {it} iters | SSE={sse:.4f} | converged={converged}")

            if sse < best_sse:
                best_sse = sse
                best = (centroids.copy(), labels.copy(), it)

        # commit best
        self.cluster_centers_ = best[0]
        self.labels_ = best[1]
        self.n_iter_ = int(best[2])
        self.inertia_ = float(best_sse)
        return self

    # ---------- inference ----------
    def predict(self, X):
        """
        [Problem 7] Assign new data to nearest learned centroids.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit(X) before predict(X).")
        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X, self.cluster_centers_)

    # ---------- elbow helper ----------
    def elbow_curve(self, X, k_values, n_init=None, max_iter=None, tol=None, verbose=None):
        """
        [Problem 8] Return list of SSE values for each k in k_values.
        """
        X = np.asarray(X, dtype=float)
        sse_list = []
        for k in k_values:
            km = ScratchKMeans(
                n_clusters=k,
                n_init=self.n_init if n_init is None else n_init,
                max_iter=self.max_iter if max_iter is None else max_iter,
                tol=self.tol if tol is None else tol,
                verbose=self.verbose if verbose is None else verbose,
            ).fit(X)
            sse_list.append(km.inertia_)
        return sse_list


# =========================================================
# Silhouette (Problem 9)
# =========================================================

def silhouette_scores(X, labels):
    """
    Compute per-sample silhouette values and the average silhouette.

    s_i = (b_i - a_i) / max(a_i, b_i)
      a_i: mean distance to points in same cluster
      b_i: min over other clusters of mean distance to that cluster

    Returns
    -------
    sil_vals : (n,) silhouette per point
    sil_avg  : float
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = X.shape[0]
    uniq = np.unique(labels)
    K = len(uniq)

    # Precompute pairwise distances
    # (simple, O(n^2). For bigger n, consider chunking.)
    dmat = np.sqrt(ScratchKMeans._euclidean_sq(X, X))  # (n,n)

    sil = np.zeros(n, dtype=float)

    for idx in range(n):
        c = labels[idx]
        in_c = (labels == c)
        out_c = (labels != c)

        # a_i: mean distance to same cluster (exclude self)
        same = np.where(in_c)[0]
        if same.size <= 1:
            a_i = 0.0
        else:
            a_i = np.mean(dmat[idx, same[same != idx]])

        # b_i: min mean distance to another cluster
        b_i = np.inf
        for c2 in uniq:
            if c2 == c:
                continue
            in_c2 = (labels == c2)
            if np.any(in_c2):
                b_i = min(b_i, np.mean(dmat[idx, in_c2]))

        denom = max(a_i, b_i) if max(a_i, b_i) > 0 else 1.0
        sil[idx] = (b_i - a_i) / denom

    return sil, float(np.mean(sil))


def plot_silhouette_diagram(sil_vals, sil_avg, labels):
    """
    Plot the silhouette diagram as requested (Problem 9).
    """
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    cluster_labels = np.arange(n_clusters)

    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = sil_vals[labels == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        plt.barh(
            range(y_ax_lower, y_ax_upper),
            c_silhouette_vals,
            height=1.0,
            edgecolor="none",
            color=color,
        )
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    plt.axvline(sil_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel("Cluster")
    plt.xlabel("Silhouette coefficient")
    plt.title("Silhouette diagram")
    plt.grid(True, axis="x", alpha=0.3)
    plt.show()


# =========================================================
# Demos — Simple dataset 3 (make_blobs)
# =========================================================

def demo_blobs():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(
        n_samples=300,
        n_features=2,
        centers=4,
        cluster_std=0.6,
        shuffle=True,
        random_state=0,
    )

    # Try k=4
    km = ScratchKMeans(n_clusters=4, n_init=10, max_iter=300, tol=1e-4, verbose=True)
    km.fit(X)
    print(f"SSE (inertia): {km.inertia_:.2f} | iters: {km.n_iter_}")

    # Plot result
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, s=25, edgecolor="k")
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="red", s=120, marker="X", label="centers")
    plt.title("K-means (scratch) on blobs, k=4")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Elbow: try k=2..8
    ks = list(range(2, 9))
    sse_list = km.elbow_curve(X, ks)
    plt.figure()
    plt.plot(ks, sse_list, marker="o")
    plt.xlabel("k")
    plt.ylabel("SSE (inertia)")
    plt.title("Elbow curve")
    plt.grid(True)
    plt.show()

    # Silhouette
    sil_vals, sil_avg = silhouette_scores(X, km.labels_)
    print(f"Average silhouette: {sil_avg:.3f}")
    plot_silhouette_diagram(sil_vals, sil_avg, km.labels_)


# =========================================================
# Wholesale customers workflow (Problems 10–12)
# =========================================================

def wholesale_workflow(csv_path, k=3, log_transform=True, show_pca=True):
    """
    Load Wholesale customers data, run K-means, visualize (PCA 2D), and show cluster summaries.

    csv_path : str to 'Wholesale customers data.csv'
    k : int, number of clusters
    log_transform : bool, log1p transform (helps with skew/heavy tails)
    show_pca : bool, require sklearn PCA to plot 2D
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    # Monetary columns (as in the UCI dataset)
    money_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    # Some files spell 'Delicatessen' — try to handle both
    if "Delicatessen" in df.columns:
        money_cols[-1] = "Delicatessen"

    data = df[money_cols].astype(float).values

    if log_transform:
        data = np.log1p(data)

    # Pick k via elbow/silhouette separately if needed.
    km = ScratchKMeans(n_clusters=k, n_init=10, max_iter=300, tol=1e-4, verbose=True).fit(data)

    print(f"[Wholesale] k={k} | SSE={km.inertia_:.2f} | iters={km.n_iter_}")

    # PCA for 2D plot
    if show_pca:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2).fit(data)
            data_pca = pca.transform(data)

            plt.figure()
            plt.scatter(data_pca[:, 0], data_pca[:, 1], c=km.labels_, s=25, edgecolor="k")
            centers_pca = pca.transform(km.cluster_centers_)
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", s=120, marker="X", label="centers")
            plt.title("Wholesale customers — PCA(2) + K-means clusters")
            plt.legend(); plt.grid(True); plt.show()

            # Explained variance chart (sample code adapted)
            pca_full = PCA(n_components=None).fit(data)
            var_exp = pca_full.explained_variance_ratio_
            cum_var_exp = np.cumsum(var_exp)
            print("Cumulative explained variance:", np.round(cum_var_exp, 3))
            plt.figure()
            plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.5, align="center", label="individual explained variance")
            plt.step(range(1, len(var_exp)+1), cum_var_exp, where="mid", label="cumulative explained variance")
            plt.ylabel("Explained variance ratio")
            plt.xlabel("Principal components")
            plt.hlines(0.7, 0, len(var_exp), colors="blue", linestyles="dashed")
            plt.legend(loc="best"); plt.grid(); plt.show()
        except Exception as e:
            print("PCA plotting skipped (sklearn not available?):", e)

    # Cluster summaries (useful for Problem 12)
    try:
        import pandas as pd
        df_clusters = pd.DataFrame(data, columns=[f"log1p_{c}" if log_transform else c for c in money_cols])
        df_clusters["cluster"] = km.labels_
        print("\nCluster summary (means):")
        print(df_clusters.groupby("cluster").mean().round(2))
        print("\nCluster sizes:")
        print(df_clusters["cluster"].value_counts().sort_index())
    except Exception as e:
        print("Pandas summary skipped:", e)

    # Optional: compare clusters with known groups (Region, Channel) if present
    if {"Region", "Channel"}.issubset(df.columns):
        df_tmp = df.copy()
        df_tmp["cluster"] = km.labels_
        for col in ["Region", "Channel"]:
            print(f"\nCrosstab cluster vs {col}:")
            print(pd.crosstab(df_tmp["cluster"], df_tmp[col]))


# =========================================================
# Advanced (Problems 13–14): t-SNE + DBSCAN (sklearn)
# =========================================================

def tsne_dbscan_compare(csv_path, eps=2.0, min_samples=5, log_transform=True):
    """
    Reduce Wholesale customers data with t-SNE, then cluster with DBSCAN.
    Compare visually with PCA + K-means (qualitative).

    Requires scikit-learn.
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import DBSCAN

    df = pd.read_csv(csv_path)
    money_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    if "Delicatessen" in df.columns:
        money_cols[-1] = "Delicatessen"
    data = df[money_cols].astype(float).values
    if log_transform:
        data = np.log1p(data)

    # t-SNE (2D)
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30.0, random_state=0)
    data_tsne = tsne.fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_tsne)
    labels_db = db.labels_

    plt.figure()
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels_db, s=25, edgecolor="k")
    plt.title("Wholesale — t-SNE (2D) + DBSCAN labels")
    plt.grid(True); plt.show()

    # Reference PCA + KMeans
    from sklearn.cluster import KMeans
    pca = PCA(n_components=2).fit(data)
    data_pca = pca.transform(data)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(data)
    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans.labels_, s=25, edgecolor="k")
    plt.title("Wholesale — PCA (2D) + KMeans labels")
    plt.grid(True); plt.show()

    # Quick note on pros/cons (Problem 13)
    print("\nDBSCAN pros: finds arbitrary-shaped clusters; handles noise; no need to prechoose K.")
    print("DBSCAN cons: eps/min_samples sensitive; struggles with varying densities; scaling to high-dim is tricky.")
    print("t-SNE pros: preserves local structure; great 2D/3D visualization.")
    print("t-SNE cons: non-parametric (no transform for new points), stochastic, parameter-sensitive, slow for large n.")


# =========================================================
# Run small demos if executed directly
# =========================================================

if __name__ == "__main__":
    # Demo on blobs
    demo_blobs()

   
