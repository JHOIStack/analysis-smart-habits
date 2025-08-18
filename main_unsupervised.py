# main_unsupervised.py
import pandas as pd
from sklearn.cluster import KMeans
from db.database import SessionLocal
from db.models import User, UserHabit, Habit, Interaction
import json

def load_data():
    session = SessionLocal()

    users = session.query(User.id, User.age, User.region).all()
    df_users = pd.DataFrame(users, columns=["userId", "age", "region"])

    habits = session.query(UserHabit.userId, Habit.category).join(Habit, Habit.id == UserHabit.habitId).all()
    df_habits = pd.DataFrame(habits, columns=["userId", "category"])
    df_habits = pd.get_dummies(df_habits, columns=["category"]).groupby("userId").sum().reset_index()

    inters = session.query(Interaction.userId, Interaction.type).all()
    df_inter = pd.DataFrame(inters, columns=["userId", "type"])
    df_inter = pd.get_dummies(df_inter, columns=["type"]).groupby("userId").sum().reset_index()

    session.close()


    df = df_users.merge(df_habits, on="userId", how="left")
    df = df.merge(df_inter, on="userId", how="left")
    df = df.fillna(0)
    return df

def clustering(df, n_clusters=3):
    X = df.drop(columns=["userId", "region"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)
    return df, kmeans

def main():
    df = load_data()
    if df.empty:
        print("No hay datos suficientes para clustering.")
        return

    df, model = clustering(df, n_clusters=3)


    results = []
    for _, row in df.iterrows():
        user_json = {
            "userId": row["userId"],
            "cluster": int(row["cluster"]),
            "features": {
                "age": {"value": row["age"], "description": "Edad del usuario"},
                "habits": {col: {"value": row[col], "description": f"Numero de habitos de tipo {col.split('_')[1]}"} 
                           for col in df.columns if col.startswith("category_")},
                "interactions": {col: {"value": row[col], "description": f"Numero de interacciones tipo {col.split('_')[1]}"} 
                                 for col in df.columns if col.startswith("type_")}
            }
        }
        results.append(user_json)

    print(json.dumps(results, indent=2))
# Función para FastAPI
def cluster_profiles():
    df = load_data()
    if df.empty:
        return []
    df, model = clustering(df, n_clusters=3)
    results = []
    for _, row in df.iterrows():
        user_json = {
            "userId": row["userId"],
            "cluster": int(row["cluster"]),
            "features": {
                "age": {"value": row["age"], "description": "Edad del usuario"},
                "habits": {col: {"value": row[col], "description": f"Numero de habitos de tipo {col.split('_')[1]}"} 
                           for col in df.columns if col.startswith("category_")},
                "interactions": {col: {"value": row[col], "description": f"Numero de interacciones tipo {col.split('_')[1]}"} 
                                 for col in df.columns if col.startswith("type_")}
            }
        }
        results.append(user_json)
    return results
