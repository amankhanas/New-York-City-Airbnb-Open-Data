import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv("data/raw/AB_NYC_2019.csv")

print("Исходный размер:", df.shape)

# Удаление дубликатов
df = df.drop_duplicates()

# Обработка пропусков
df["name"] = df["name"].fillna("Unknown")
df["host_name"] = df["host_name"].fillna("Unknown")
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

# Удаление строк без координат
df = df.dropna(subset=["latitude", "longitude"])

# Удаление экстремальных выбросов цены
q1 = df["price"].quantile(0.25)
q3 = df["price"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df[(df["price"] >= lower) & (df["price"] <= upper)]

print("После очистки:", df.shape)

# Сохранение
df.to_csv("data/processed/airbnb_clean.csv", index=False)

print("Файл airbnb_clean.csv сохранён")
