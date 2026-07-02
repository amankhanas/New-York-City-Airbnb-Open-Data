from pathlib import Path
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "processed" / "airbnb_clean.csv"
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    numeric = df.select_dtypes(include=["number"])

    desc = numeric.describe().T
    desc["variance"] = numeric.var()
    desc["std_dev"] = numeric.std()
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["skewness"] = numeric.skew()
    desc["kurtosis"] = numeric.kurt()

    corr = numeric.corr()
    target_corr = corr["price"].sort_values(ascending=False).to_frame(name="correlation_with_price")

    price_q1 = df["price"].quantile(0.25)
    price_q3 = df["price"].quantile(0.75)
    price_iqr = price_q3 - price_q1
    lower_bound = price_q1 - 1.5 * price_iqr
    upper_bound = price_q3 + 1.5 * price_iqr
    outlier_count = int(((df["price"] < lower_bound) | (df["price"] > upper_bound)).sum())

    z_scores = (df["price"] - df["price"].mean()) / df["price"].std(ddof=0)
    z_outlier_count = int((z_scores.abs() > 3).sum())

    outlier_summary = pd.DataFrame(
        {
            "metric": [
                "iqr_lower_bound",
                "iqr_upper_bound",
                "iqr_outlier_count",
                "zscore_threshold",
                "zscore_outlier_count",
            ],
            "value": [lower_bound, upper_bound, outlier_count, 3, z_outlier_count],
        }
    )

    print("Описание числовых признаков:\n")
    print(desc)
    print("\nКорреляция признаков с ценой:\n")
    print(target_corr)

    desc.to_csv(output_dir / "descriptive_statistics.csv")
    corr.to_csv(output_dir / "correlation_matrix.csv")
    target_corr.to_csv(output_dir / "price_correlation.csv")
    outlier_summary.to_csv(output_dir / "outlier_summary.csv", index=False)

    print("Файлы сохранены в data/processed")


if __name__ == "__main__":
    main()
