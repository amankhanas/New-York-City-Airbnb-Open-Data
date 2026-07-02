import pandas as pd  # импортируем pandas для статистической обработки данных


def main():
    df = pd.read_csv("data/processed/airbnb_clean.csv")  # читаем очищенный набор данных Airbnb

    numeric = df.select_dtypes(include=["int64", "float64"])  # выбираем только числовые признаки
    desc = numeric.describe().T  # получаем статистическое описание для каждого числового признака
    corr = numeric.corr()  # вычисляем корреляционную матрицу
    target_corr = corr["price"].sort_values(ascending=False)  # сортируем корреляцию с ценой по величине

    print("Описание числовых признаков:\n")  # выводим заголовок для описательных статистик
    print(desc)  # печатаем описательное статистическое резюме
    print("\nКорреляция признаков с ценой:\n")  # выводим заголовок для корреляции с ценой
    print(target_corr)  # печатаем корреляции с целевой переменной

    desc.to_csv("data/processed/descriptive_statistics.csv")  # сохраняем описательную статистику в CSV
    corr.to_csv("data/processed/correlation_matrix.csv")  # сохраняем корреляционную матрицу в CSV
    target_corr.to_csv("data/processed/price_correlation.csv")  # сохраняем корреляции с ценой в CSV

    print("Файлы сохранены в data/processed: descriptive_statistics.csv, correlation_matrix.csv, price_correlation.csv")  # уведомляем об успешном сохранении файлов


if __name__ == "__main__":
    main()  # запускаем основную функцию при выполнении скрипта напрямую
