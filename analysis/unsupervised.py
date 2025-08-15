import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.dataset import build_dataset

def cluster_users(n_clusters=4):
    df = build_dataset()
    
    features = df.select_dtypes(include=["int64", "float64"])
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    result = df[["id", "name", "age", "region", "cluster"]]
    return result
