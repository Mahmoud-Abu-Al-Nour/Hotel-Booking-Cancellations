from src.data_cleaning import *
from src.preprocessing import *
from src.train_model import *
from src.utils import *

def main():

    df = load_data(
        "data/raw/HotelData.csv"
    )

    df = create_features(df)

    df = encode_target(df)

    save_cleaned_data(
        df,
        "data/processed/hotel_bookings_cleaned.csv"
    )

    X, y = prepare_features(df)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    model = train_gradient_boosting(
        X_train,
        y_train
    )

    score = evaluate_model(
        model,
        X_test,
        y_test
    )

    print(
        f"Accuracy: {score:.4f}"
    )

    save_model(
        model,
        "models/gb_booking_model.pkl"
    )

if __name__ == "__main__":
    main()