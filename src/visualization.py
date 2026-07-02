import pandas as pd  # импортируем pandas для загрузки и обработки данных
import matplotlib.pyplot as plt  # импортируем matplotlib для построения графиков
import seaborn as sns  # импортируем seaborn для стилизованных визуализаций

try:
    import missingno as msno  # пытаемся импортировать missingno для визуализации пропущенных значений
except ImportError:
    msno = None  # если missingno не установлен, то проигнорируем матрицу пропусков
    print("WARNING: missingno is not installed. Missing values plot will be skipped.")  # предупреждение об отсутствии optional-библиотеки

df = pd.read_csv("data/processed/airbnb_clean.csv")  # читаем очищенный набор данных Airbnb из CSV

if msno is not None:  # проверяем, можно ли строить график пропусков
    msno.matrix(df)  # рисуем матрицу пропусков
    plt.savefig("img/missing_values.png")  # сохраняем результат в файл
    plt.close()  # закрываем фигуру, освобождаем ресурсы
else:
    print("Skipping missing values chart because missingno is unavailable.")  # уведомляем, что график пропусков пропущен

plt.figure(figsize=(10,6))  # создаём фигуру для гистограммы цен
sns.histplot(df["price"], bins=50)  # строим гистограмму распределения цен
plt.title("Price Distribution")  # задаём заголовок графика
plt.savefig("img/price_distribution.png")  # сохраняем гистограмму в папку img
plt.close()  # закрываем фигуру

plt.figure(figsize=(8,5))  # создаём фигуру для столбчатой диаграммы по районам

df.groupby("neighbourhood_group")["price"].mean().sort_values().plot(kind="bar")  # строим бар-чарт средней цены по району
plt.title("Average Price by Borough")  # заголовок диаграммы
plt.savefig("img/borough_price.png")  # сохраняем диаграмму
plt.close()  # закрываем фигуру

plt.figure(figsize=(8,5))  # создаём фигуру для boxplot
sns.boxplot(data=df, x="room_type", y="price")  # строим boxplot цен для каждого типа жилья
plt.xticks(rotation=20)  # поворачиваем подписи, чтобы они не перекрывались
plt.savefig("img/room_type_boxplot.png")  # сохраняем boxplot
plt.close()  # закрываем фигуру

plt.figure(figsize=(8,5))  # создаём фигуру для scatter plot
sns.scatterplot(
    data=df,  # передаём DataFrame
    x="number_of_reviews",  # ось X — количество отзывов
    y="price"  # ось Y — цена
)
plt.savefig("img/scatter_reviews_price.png")  # сохраняем scatter plot
plt.close()  # закрываем фигуру

plt.figure(figsize=(10,8))  # создаём фигуру для корреляционной матрицы
corr = df.select_dtypes(include=["int64","float64"]).corr()  # вычисляем корреляционную матрицу для числовых признаков
sns.heatmap(
    corr,  # передаём матрицу корреляций
    annot=True,  # включаем числовые аннотации
    cmap="coolwarm"  # используем цветовую карту
)
plt.savefig("img/correlation_matrix.png")  # сохраняем тепловую карту
plt.close()  # закрываем фигуру

print("Все графики сохранены в папку img")  # выводим сообщение об успешном сохранении графиков