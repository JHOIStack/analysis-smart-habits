import requests
import pandas as pd
import plotly.express as px
import imageio.v2 as imageio
import numpy as np

# --- Fetch API ---
unsupervised = requests.get("http://localhost:8000/ml/unsupervised").json()

# --- Preparar datos ---
scatter_data = []
for u in unsupervised:
    scatter_data.append({
        "userId": u["userId"],
        "cluster": u["cluster"],
        "age": u["features"]["age"]["value"],
        "total_habits": sum([v["value"] for v in u["features"]["habits"].values()]),
        "total_interactions": sum([v["value"] for v in u["features"]["interactions"].values()])
    })
df = pd.DataFrame(scatter_data)

# --- Crear figura inicial ---
fig = px.scatter_3d(
    df,
    x="age",
    y="total_habits",
    z="total_interactions",
    color="cluster",
    hover_data=["userId"],
    size="total_habits",
    size_max=20
)

fig.update_layout(
    scene=dict(
        xaxis_title='Edad',
        yaxis_title='Total Hábitos',
        zaxis_title='Total Interacciones',
    ),
    paper_bgcolor='rgba(20,20,30,1)',
    font=dict(color='white')
)

# --- Animar la cámara (versión rápida) ---
frames = []
num_frames = 120  # 2s a 60fps o suficiente para GIF
radius = 1.25
for i in range(num_frames):
    angle = i * 2 * np.pi / num_frames
    camera = dict(
        eye=dict(x=radius*np.cos(angle), y=radius*np.sin(angle), z=0.8)
    )
    fig.update_layout(scene_camera=camera)
    img_bytes = fig.to_image(format="png", width=600, height=450)
    frames.append(imageio.imread(img_bytes))

# --- Guardar GIF ---
imageio.mimsave("scatter3d_animation.gif", frames, fps=30)
print("GIF generado: scatter3d_animation.gif")
