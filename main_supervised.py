import json
import pandas as pd
from db.database import SessionLocal
from db.models import User, Profile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def fetch_data():
    session = SessionLocal()
    try:
        # Traemos usuarios junto con sus perfiles
        data = (
            session.query(User.id, User.age, User.region, Profile.profileType)
            .join(Profile, Profile.userId == User.id)
            .all()
        )
    finally:
        session.close()

    # Convertimos a DataFrame
    df = pd.DataFrame(data, columns=["id", "age", "region", "profileType"])
    return df


def run_model():
    df = fetch_data()

    df = df.dropna(subset=["profileType", "age", "region"])

    le_region = LabelEncoder()
    df["region_encoded"] = le_region.fit_transform(df["region"].astype(str))

    le_profile = LabelEncoder()
    df["profile_encoded"] = le_profile.fit_transform(df["profileType"].astype(str))

    X = df[["age", "region_encoded"]]
    y = df["profile_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Modelo
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predicciones
    y_pred = clf.predict(X_test)

    # Reporte
    report = classification_report(
        y_test, y_pred, target_names=le_profile.classes_, output_dict=True
    )

    # Resultados JSON-friendly
    results = []
    for idx, user_id in enumerate(df.iloc[y_test.index]["id"].values):
        results.append({
            "userId": user_id,
            "realProfile": le_profile.inverse_transform([y_test.iloc[idx]])[0],
            "predictedProfile": le_profile.inverse_transform([y_pred[idx]])[0],
        })

    output = {
        "classification_report": report,
        "predictions": results
    }

    return output


# Función para FastAPI
def classify_profiles():
    return run_model()
