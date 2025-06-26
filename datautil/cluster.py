import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def automated_k_estimation(features, k_min=2, k_max=10):
    """
    Automatically estimates the best K using silhouette scoring.

    Args:
        features (np.array): Feature embeddings from DIVERSIFY encoder.
        k_min (int): Minimum number of clusters.
        k_max (int): Maximum number of clusters.

    Returns:
        int: Optimal number of clusters (K).
    """
    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
        labels = kmeans.labels_
        score = silhouette_score(features, labels)
        
        if score > best_score:
            best_k = k
            best_score = score
    
    print(f"Automated K-Estimation found best K={best_k} with silhouette score={best_score:.4f}")
    return best_k
