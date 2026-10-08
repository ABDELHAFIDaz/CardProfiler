import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from cardprofiler.preprocessing import COLS


def kmeans_grid_search(df_prep, comps=range(2, 6), ks=range(2, 8)):
    results = []
    for n_comp in comps:
        X_pca = PCA(n_components=n_comp).fit_transform(df_prep[COLS])
        for k in ks:
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = model.fit_predict(X_pca)
            if len(np.unique(labels)) < 2:
                continue
            results.append({
                "n_components": n_comp,
                "k": k,
                "silhouette": silhouette_score(X_pca, labels),
                "inertia": model.inertia_
            })
    return pd.DataFrame(results).sort_values("silhouette", ascending=False).reset_index(drop=True)



def fit_kmeans(df_prep, n_components, k):
    X_pca = PCA(n_components=n_components).fit_transform(df_prep[COLS])
    model = KMeans(n_clusters=k , random_state=42, n_init=20)
    labels = model.fit_predict(X_pca)
    return labels, X_pca, model



CLUSTER_NAMES = {0: "Clients à Risque", 1: "Clients Premium", 2: "Clients à Faible Activité"}


def add_target(df_clean, labels):
    df = df_clean.copy()
    df["cluster_final"] = labels
    df["target"] = df["cluster_final"].map(CLUSTER_NAMES)
    assert df["target"].isnull().sum() == 0
    return df