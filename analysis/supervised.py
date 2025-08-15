import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from utils.dataset import build_dataset

def classify_profiles():
    df = build_dataset(include_profiles=True)

    df = df.dropna(subset=["profileType"])
    
    X = df.select_dtypes(include=["int64", "float64"])
    y = df["profileType"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_real_labels = le.inverse_transform(y_test)

    result = pd.DataFrame({
        "userId": df.iloc[y_test.index]["id"].values,
        "realProfile": y_real_labels,
        "predictedProfile": y_pred_labels,
    })

    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return result
