from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import missingno as msno
except ImportError:
    msno = None
    print("WARNING: missingno is not installed. Missing values plot will be skipped.")

repo_root = Path(__file__).resolve().parents[1]
img_dir = repo_root / "img"
img_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(repo_root / "data" / "processed" / "airbnb_clean.csv")

if msno is not None:
    msno.matrix(df)
    plt.savefig(img_dir / "missing_values.png")
    plt.close()
else:
    print("Skipping missing values chart because missingno is unavailable.")

plt.figure(figsize=(10, 6))
sns.histplot(df["price"], bins=50)
plt.title("Price Distribution")
plt.tight_layout()
plt.savefig(img_dir / "price_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
df.groupby("neighbourhood_group")["price"].mean().sort_values().plot(kind="bar")
plt.title("Average Price by Borough")
plt.tight_layout()
plt.savefig(img_dir / "borough_price.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="room_type", y="price")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(img_dir / "room_type_boxplot.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="number_of_reviews", y="price")
plt.title("Price vs Number of Reviews")
plt.tight_layout()
plt.savefig(img_dir / "scatter_reviews_price.png")
plt.close()

plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(img_dir / "correlation_matrix.png")
plt.close()

print("Все графики сохранены в папку img")
