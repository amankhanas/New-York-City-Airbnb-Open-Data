import pandas as pd


def main():
    df = pd.read_csv("data/processed/airbnb_clean.csv")

    numeric = df.select_dtypes(include=["int64", "float64"])
    desc = numeric.describe().T
    corr = numeric.corr()
    target_corr = corr["price"].sort_values(ascending=False)

    print("Описание числовых признаков:\n")
    print(desc)
    print("\nКорреляция признаков с ценой:\n")
    print(target_corr)

    desc.to_csv("data/processed/descriptive_statistics.csv")
    corr.to_csv("data/processed/correlation_matrix.csv")
    target_corr.to_csv("data/processed/price_correlation.csv")

    print("Файлы сохранены в data/processed: descriptive_statistics.csv, correlation_matrix.csv, price_correlation.csv")


if __name__ == "__main__":
    main()
