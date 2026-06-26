import pandas as pd


def main():
    df = pd.read_csv("data/raw/AB_NYC_2019.csv")

    print("Исходный размер данных:", df.shape)

    # Удаляем полные дубликаты
    df = df.drop_duplicates()

    # Обрабатываем пропуски
    df["name"] = df["name"].fillna("Unknown")
    df["host_name"] = df["host_name"].fillna("Unknown")
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Удаляем строки без геокоординат
    df = df.dropna(subset=["latitude", "longitude"])

    # Удаляем некорректные и экстремальные значения цены
    df = df[(df["price"] > 0) & (df["price"] <= 1000)]

    # Промежуточная фильтрация по индентификатору и ночам
    df = df[df["minimum_nights"] >= 1]

    # Удаление выбросов по цене через IQR
    q1 = df["price"].quantile(0.25)
    q3 = df["price"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]

    print("Размер после очистки:", df.shape)

    df.to_csv("data/processed/airbnb_clean.csv", index=False)
    print("Очищенный файл сохранён: data/processed/airbnb_clean.csv")


if __name__ == "__main__":
    main()
