import pandas as pd  # импортируем pandas для работы с табличными данными


def main():
    df = pd.read_csv("data/raw/AB_NYC_2019.csv")  # читаем исходный датасет Airbnb из папки data/raw

    print("Исходный размер данных:", df.shape)  # выводим размер исходного набора данных

    df = df.drop_duplicates()  # удаляем полностью повторяющиеся строки

    df["name"] = df["name"].fillna("Unknown")  # заменяем пустые названия объектов на Unknown
    df["host_name"] = df["host_name"].fillna("Unknown")  # заменяем пустые имена хостов на Unknown
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)  # пропуски в reviews_per_month заменяем на 0
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")  # преобразуем дату последнего отзыва в datetime

    df = df.dropna(subset=["latitude", "longitude"])  # удаляем строки без координат

    df = df[(df["price"] > 0) & (df["price"] <= 1000)]  # отбрасываем некорректные и экстремальные цены

    df = df[df["minimum_nights"] >= 1]  # оставляем только записи с минимум одной ночи

    q1 = df["price"].quantile(0.25)  # вычисляем первый квартиль цены
    q3 = df["price"].quantile(0.75)  # вычисляем третий квартиль цены
    iqr = q3 - q1  # интерквартильный размах
    lower = q1 - 1.5 * iqr  # нижняя граница выбросов по цене
    upper = q3 + 1.5 * iqr  # верхняя граница выбросов по цене
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]  # удаляем выбросы по цене

    print("Размер после очистки:", df.shape)  # выводим размер данных после очистки

    df.to_csv("data/processed/airbnb_clean.csv", index=False)  # сохраняем очищенный набор данных в CSV
    print("Очищенный файл сохранён: data/processed/airbnb_clean.csv")  # уведомляем об успешном сохранении


if __name__ == "__main__":
    main()  # запускаем функцию main, если скрипт выполнен напрямую
