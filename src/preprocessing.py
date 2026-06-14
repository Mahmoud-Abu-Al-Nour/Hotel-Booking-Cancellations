FEATURES = [
    "lead_time",
    "avg_price_per_room",
    "no_of_special_requests",
    "total_guests",
    "total_nights",
    "repeated_guest"
]


def prepare_features(df):

    X = df[FEATURES]

    y = df["booking_status"]

    return X, y