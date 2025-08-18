import requests
import pandas as pd
import plotly.express as px

# --- Fetch API ---
users = requests.get("http://localhost:3000/api/users").json()
clusters = requests.get("http://localhost:8000/ml/unsupervised").json()

# --- Preparar DataFrames ---
df_users = pd.DataFrame([{
    "userId": u["id"],
    "region": u["region"],
    "age": u["age"]
} for u in users])

df_clusters = pd.DataFrame([{
    "userId": c["userId"],
    "cluster": c["cluster"],
    "total_habits": sum([v["value"] for v in c["features"]["habits"].values()]),
    "total_interactions": sum([v["value"] for v in c["features"]["interactions"].values()])
} for c in clusters])

df = pd.merge(df_users, df_clusters, on="userId")

region_coords = {
    "NORTE": {"lat": 26, "lon": -106},
    "CENTRO": {"lat": 21, "lon": -100},
    "SUR": {"lat": 16, "lon": -92},
    "OCCIDENTE": {"lat": 20, "lon": -103},
    "SURESTE": {"lat": 17, "lon": -89},
    "CDMX": {"lat": 19.43, "lon": -99.13},
    "INTERNACIONAL": {"lat": 0, "lon": 0}
}

df["lat"] = df["region"].map(lambda r: region_coords[r]["lat"])
df["lon"] = df["region"].map(lambda r: region_coords[r]["lon"])

# --- Configurar Mapbox ---
px.set_mapbox_access_token("pk.eyJ1Ijoib2N0YXZpb2RldnRlY2giLCJhIjoiY21lZ243aXRoMTdtZTJtcHhtOXkyNHc2diJ9.UNuvyMMXK3FnPHN9SbtELw")

# --- Paleta formal ---
cluster_colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e"}  # azul, verde, naranja

fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="cluster",
    color_discrete_map=cluster_colors,
    size="total_habits",
    size_max=18,
    opacity=0.75,
    hover_name="userId",
    hover_data={
        "region": True,
        "age": True,
        "total_habits": True,
        "total_interactions": True,
        "lat": False,
        "lon": False
    },
    zoom=3.5,
    height=700,
    mapbox_style="light",
    title="Distribución de Usuarios por Región y Cluster"
)

# --- Layout formal ---
fig.update_layout(
    mapbox=dict(
        center=dict(lat=19.43, lon=-99.13),  # CDMX como centro inicial
        zoom=5,                              # zoom inicial
        style="light"
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)


fig.show()
