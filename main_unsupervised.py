
"""
main_unsupervised.py
-----------------------------------
Script principal para análisis no supervisado (clustering) de usuarios.
Utiliza KMeans para agrupar usuarios según edad, hábitos e interacciones.
Incluye funciones para uso directo y para integración con FastAPI.
"""

import pandas as pd
from sklearn.cluster import KMeans
from db.database import SessionLocal
from db.models import User, UserHabit, Habit, Interaction
import json


def load_data():
    """
    Carga los datos de usuarios, hábitos e interacciones desde la base de datos.
    Devuelve un DataFrame con todas las características necesarias para clustering.
    """
    session = SessionLocal()
    try:
        # Usuarios
        users = session.query(User.id, User.age, User.region).all()
        df_users = pd.DataFrame(users, columns=["userId", "age", "region"])

        # Hábitos
        habits = (
            session.query(UserHabit.userId, Habit.category)
            .join(Habit, Habit.id == UserHabit.habitId)
            .all()
        )
        df_habits = pd.DataFrame(habits, columns=["userId", "category"])
        df_habits = pd.get_dummies(df_habits, columns=["category"]).groupby("userId").sum().reset_index()

        # Interacciones
        inters = session.query(Interaction.userId, Interaction.type).all()
        df_inter = pd.DataFrame(inters, columns=["userId", "type"])
        df_inter = pd.get_dummies(df_inter, columns=["type"]).groupby("userId").sum().reset_index()
    finally:
        session.close()

    # Unir todo en un solo DataFrame
    df = df_users.merge(df_habits, on="userId", how="left")
    df = df.merge(df_inter, on="userId", how="left")
    df = df.fillna(0)
    return df


def clustering(df, n_clusters=3):
    """
    Aplica KMeans para agrupar usuarios en n_clusters.
    Devuelve el DataFrame con la columna 'cluster' y el modelo entrenado.
    """
    features = df.drop(columns=["userId", "region"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(features)
    return df, kmeans

# --- Ejecución directa ---
def main():
    """
    Ejecuta el proceso de clustering y muestra los resultados en formato JSON.
    """
    df = load_data()
    if df.empty:
        print("No hay datos suficientes para clustering.")
        return

    df, _ = clustering(df, n_clusters=3)

    results = []
    for _, row in df.iterrows():
        user_json = {
            "userId": row["userId"],
            "cluster": int(row["cluster"]),
            "features": {
                "age": {"value": row["age"], "description": "Edad del usuario"},
                "habits": {
                    col: {
                        "value": row[col],
                        "description": f"Numero de habitos de tipo {col.split('_')[1]}"
                    }
                    for col in df.columns if col.startswith("category_")
                },
                "interactions": {
                    col: {
                        "value": row[col],
                        "description": f"Numero de interacciones tipo {col.split('_')[1]}"
                    }
                    for col in df.columns if col.startswith("type_")
                }
            }
        }
        results.append(user_json)

    print(json.dumps(results, indent=2))

# --- API para FastAPI ---
def cluster_profiles():
    """
    Función para uso con FastAPI. Devuelve los resultados de clustering en formato lista de dicts.
    """
    df = load_data()
    if df.empty:
        return []
    df, _ = clustering(df, n_clusters=3)
    results = []
    for _, row in df.iterrows():
        user_json = {
            "userId": row["userId"],
            "cluster": int(row["cluster"]),
            "features": {
                "age": {"value": row["age"], "description": "Edad del usuario"},
                "habits": {
                    col: {
                        "value": row[col],
                        "description": f"Numero de habitos de tipo {col.split('_')[1]}"
                    }
                    for col in df.columns if col.startswith("category_")
                },
                "interactions": {
                    col: {
                        "value": row[col],
                        "description": f"Numero de interacciones tipo {col.split('_')[1]}"
                    }
                    for col in df.columns if col.startswith("type_")
                }
            }
        }
        results.append(user_json)
    return results
