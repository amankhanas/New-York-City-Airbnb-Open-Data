from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "processed" / "airbnb_clean.csv"
    output_dir = repo_root / "data" / "processed"
    img_dir = repo_root / "img"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    numeric_features = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
        "calculated_host_listings_count",
    ]
    categorical_features = ["neighbourhood_group", "room_type"]

    df_model = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    feature_columns = numeric_features + [
        col for col in df_model.columns
        if col.startswith("neighbourhood_group_") or col.startswith("room_type_")
    ]

    X = df_model[feature_columns]
    y = df_model["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = pd.DataFrame(
        [{"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "model": "LinearRegression"}]
    )
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)

    coefficients = pd.DataFrame(
        {"feature": feature_columns, "coefficient": model.coef_}
    ).sort_values(by="coefficient", key=abs, ascending=False)
    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False)

    predictions = pd.DataFrame(
        {
            "actual_price": y_test.values,
            "predicted_price": y_pred,
            "residual": y_test.values - y_pred,
        }
    )
    predictions.to_csv(output_dir / "model_predictions.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price")
    plt.savefig(img_dir / "actual_vs_predicted.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=predictions["residual"], alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig(img_dir / "residual_plot.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    top_features = coefficients.head(10).copy()
    sns.barplot(data=top_features, x="coefficient", y="feature", orient="h")
    plt.title("Top Feature Coefficients")
    plt.tight_layout()
    plt.savefig(img_dir / "feature_importance.png")
    plt.close()

    print("Метрики модели Linear Regression:\n")
    print(metrics.to_string(index=False))
    print("\nКоэффициенты модели сохранены в data/processed/model_coefficients.csv")
    print("Графики сохранены в img/")


if __name__ == "__main__":
    main()
