import pandas as pd  # импортируем pandas для подготовки данных
from sklearn.linear_model import LinearRegression  # импортируем линейную регрессию из sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # импортируем метрики для оценки модели
from sklearn.model_selection import train_test_split  # импортируем функцию разбиения данных на train/test


def main():
    df = pd.read_csv("data/processed/airbnb_clean.csv")  # читаем очищенный набор данных Airbnb

    numeric_features = [  # список числовых признаков для модели
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
        "calculated_host_listings_count",
    ]
    categorical_features = ["neighbourhood_group", "room_type"]  # категориальные признаки

    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)  # кодируем категориальные признаки dummy-переменными
    feature_columns = numeric_features + [  # формируем список признаков для обучения
        col for col in df.columns
        if col.startswith("neighbourhood_group_") or col.startswith("room_type_")
    ]

    X = df[feature_columns]  # матрица признаков
    y = df["price"]  # вектор целевой переменной — цена

    X_train, X_test, y_train, y_test = train_test_split(  # делим выборку на обучающую и тестовую
        X, y, test_size=0.2, random_state=42  # 20% данных оставляем для теста
    )

    model = LinearRegression()  # создаём объект модели линейной регрессии
    model.fit(X_train, y_train)  # обучаем модель на тренировочных данных
    y_pred = model.predict(X_test)  # делаем предсказания на тестовой выборке

    mae = mean_absolute_error(y_test, y_pred)  # считаем MAE
    mse = mean_squared_error(y_test, y_pred)  # считаем MSE
    rmse = mse ** 0.5  # считаем RMSE
    r2 = r2_score(y_test, y_pred)  # считаем R2

    print("Метрики модели Linear Regression:\n")  # выводим заголовок метрик
    print(f"MAE: {mae:.2f}")  # выводим MAE
    print(f"MSE: {mse:.2f}")  # выводим MSE
    print(f"RMSE: {rmse:.2f}")  # выводим RMSE
    print(f"R2: {r2:.3f}")  # выводим R2

    coefficients = pd.DataFrame({  # создаём таблицу коэффициентов модели
        "feature": X.columns,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", key=abs, ascending=False)  # сортируем коэффициенты по абсолютной величине
    coefficients.to_csv("data/processed/model_coefficients.csv", index=False)  # сохраняем коэффициенты в CSV
    print("Коэффициенты модели сохранены в data/processed/model_coefficients.csv")  # уведомляем об успешном сохранении


if __name__ == "__main__":
    main()  # выполняем основную функцию, если скрипт запущен напрямую
