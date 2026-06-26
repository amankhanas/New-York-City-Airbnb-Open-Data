import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("data/processed/airbnb_clean.csv")

    numeric_features = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
        "calculated_host_listings_count",
    ]
    categorical_features = ["neighbourhood_group", "room_type"]

    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    feature_columns = numeric_features + [
        col for col in df.columns
        if col.startswith("neighbourhood_group_") or col.startswith("room_type_")
    ]

    X = df[feature_columns]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("Метрики модели Linear Regression:\n")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.3f}")

    coefficients = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", key=abs, ascending=False)
    coefficients.to_csv("data/processed/model_coefficients.csv", index=False)
    print("Коэффициенты модели сохранены в data/processed/model_coefficients.csv")


if __name__ == "__main__":
    main()
