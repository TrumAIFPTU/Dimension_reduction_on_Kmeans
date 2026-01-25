from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def compute_metrics(X_emb, y_true, y_pred):
    out = {}
    out["silhouette"] = float(silhouette_score(X_emb, y_pred))
    out["ari"] = float(adjusted_rand_score(y_true, y_pred))
    out["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    return out
