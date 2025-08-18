import pandas as pd
from db.database import SessionLocal
from db import models


def get_users():
    session = SessionLocal()
    users = session.query(models.User).all()

    data = []
    for u in users:
        data.append({
            "id": u.id,
            "name": u.name,
            "age": u.age,
            "region": u.region.name if u.region else None,
            "created_at": u.createdAt,
            "role": u.role.name
        })

    session.close()
    return pd.DataFrame(data)


def get_habit_features():
    session = SessionLocal()

    data = session.query(
        models.UserHabit.userId,
        models.Habit.category
    ).join(models.Habit).all()

    session.close()

    df = pd.DataFrame(data, columns=["userId", "category"])
    features = pd.crosstab(df["userId"], df["category"])
    features.reset_index(inplace=True)
    return features


def get_interaction_features():
    session = SessionLocal()

    data = session.query(
        models.Interaction.userId,
        models.Interaction.type
    ).all()

    session.close()

    df = pd.DataFrame(data, columns=["userId", "type"])
    features = pd.crosstab(df["userId"], df["type"])
    features.reset_index(inplace=True)
    return features


def get_profiles():
    session = SessionLocal()

    data = session.query(
        models.Profile.userId,
        models.Profile.profileType
    ).all()

    session.close()

    df = pd.DataFrame(data, columns=["userId", "profileType"])
    return df


def build_dataset(include_profiles=False):
    df_users = get_users()
    df_habits = get_habit_features()
    df_interactions = get_interaction_features()

    df = df_users.merge(df_habits, left_on="id", right_on="userId", how="left")
    df = df.merge(df_interactions, on="userId", how="left")

    if include_profiles:
        df_profiles = get_profiles()
        df = df.merge(
            df_profiles,
            left_on="id",
            right_on="userId",
            how="left",
            suffixes=("", "_profile")
        )

    if "userId" in df.columns:
        df = df.drop(columns=["userId"])

    df = df.fillna(0)

    import enum
    df = df.apply(
        lambda col: col.astype(str)
        if col.dtype.name == "category" or isinstance(col.iloc[0], enum.Enum)
        else col
    )

    return df
