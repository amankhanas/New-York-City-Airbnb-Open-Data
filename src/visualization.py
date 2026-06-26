import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import missingno as msno
except ImportError:
    msno = None
    print("WARNING: missingno is not installed. Missing values plot will be skipped.")

df = pd.read_csv("data/processed/airbnb_clean.csv")

# 1. Пропущенные значения
if msno is not None:
    msno.matrix(df)
    plt.savefig("img/missing_values.png")
    plt.close()
else:
    print("Skipping missing values chart because missingno is unavailable.")

# 2. Гистограмма цен
plt.figure(figsize=(10,6))
sns.histplot(df["price"], bins=50)
plt.title("Price Distribution")
plt.savefig("img/price_distribution.png")
plt.close()

# 3. Средняя цена по районам
plt.figure(figsize=(8,5))
df.groupby("neighbourhood_group")["price"].mean().sort_values().plot(kind="bar")
plt.title("Average Price by Borough")
plt.savefig("img/borough_price.png")
plt.close()

# 4. Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="room_type", y="price")
plt.xticks(rotation=20)
plt.savefig("img/room_type_boxplot.png")
plt.close()

# 5. Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="number_of_reviews",
    y="price"
)
plt.savefig("img/scatter_reviews_price.png")
plt.close()

# 6. Correlation Matrix
plt.figure(figsize=(10,8))

corr = df.select_dtypes(include=["int64","float64"]).corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.savefig("img/correlation_matrix.png")
plt.close()

print("Все графики сохранены в папку img")