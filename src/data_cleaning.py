import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def create_features(df):

    df["total_guests"] = (
        df["no_of_adults"]
        + df["no_of_children"]
    )

    df["total_nights"] = (
        df["no_of_week_nights"]
        + df["no_of_weekend_nights"]
    )

    df["is_weekend_heavy"] = (
        df["no_of_weekend_nights"]
        > df["no_of_week_nights"]
    ).astype(int)

    return df


def encode_target(df):

    df["booking_status"] = df["booking_status"].map(
        {
            "Not_Canceled": 1,
            "Canceled": 0
        }
    )

    return df


def save_cleaned_data(df, path):

    df.to_csv(path, index=False)