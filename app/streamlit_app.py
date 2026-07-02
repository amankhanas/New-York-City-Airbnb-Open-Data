import streamlit as st  # импорт библиотеки Streamlit для создания веб-приложения
import pandas as pd  # импорт библиотеки pandas для работы с таблицами
import plotly.express as px  # импорт plotly express для интерактивных графиков
import matplotlib.pyplot as plt  # импорт matplotlib для построения визуализаций
import seaborn as sns  # импорт seaborn для тёплых карт и статистических графиков


@st.cache_data  # кэшируем данные, чтобы не загружать CSV заново при каждом обновлении
def load_data():
    return pd.read_csv("data/processed/airbnb_clean.csv")  # читаем очищенный CSV-файл Airbnb из папки data/processed


def filter_data(df):
    boroughs = st.sidebar.multiselect(  # создаём боковой фильтр районов города
        "Выберите район (borough)",  # метка для выпадающего списка районов
        options=df["neighbourhood_group"].unique(),  # уникальные значения районов из данных
        default=df["neighbourhood_group"].unique(),  # по умолчанию выбираем все районы
    )
    room_types = st.sidebar.multiselect(  # создаём боковой фильтр типа жилья
        "Выберите тип жилья",  # метка для списка типов жилья
        options=df["room_type"].unique(),  # уникальные типы жилья из данных
        default=df["room_type"].unique(),  # по умолчанию выбираем все типы жилья
    )
    price_range = st.sidebar.slider(  # создаём боковой ползунок для диапазона цен
        "Диапазон цены",  # текст для ползунка цены
        int(df["price"].min()),  # минимальное значение цены
        int(df["price"].max()),  # максимальное значение цены
        (int(df["price"].quantile(0.05)), int(df["price"].quantile(0.95))),  # диапазон по квантилям
    )

    filtered = df[  # фильтруем данные по выбранным параметрам
        (df["neighbourhood_group"].isin(boroughs)) &  # оставляем строки с выбранными районами
        (df["room_type"].isin(room_types)) &  # оставляем строки с выбранными типами жилья
        (df["price"] >= price_range[0]) &  # оставляем строки с ценой выше нижней границы
        (df["price"] <= price_range[1])  # оставляем строки с ценой ниже верхней границы
    ]
    return filtered  # возвращаем отфильтрованный DataFrame


def main():
    st.set_page_config(  # настраиваем параметры страницы Streamlit
        page_title="NYC Airbnb Dashboard",  # заголовок страницы
        layout="wide",  # широкий макет страницы
        initial_sidebar_state="expanded",  # боковая панель открыта сразу
    )

    st.title("NYC Airbnb Open Data Dashboard")  # основный заголовок приложения
    st.markdown(  # описание приложения
        "Анализ структуры, цен и отзывов Airbnb в Нью-Йорке с интерактивными фильтрами и визуализацией."
    )

    df = load_data()  # загружаем данные из CSV-файла
    filtered_df = filter_data(df)  # применяем фильтры к данным

    col1, col2, col3, col4 = st.columns(4)  # создаём 4 колонки для метрик
    col1.metric("Объявлений", len(filtered_df))  # показываем количество объявлений
    col2.metric("Медианная цена", f"${filtered_df['price'].median():.0f}")  # медианная цена
    col3.metric("Среднее число отзывов", f"{filtered_df['number_of_reviews'].mean():.1f}")  # среднее отзывы
    col4.metric("Среднее количество дней доступно", f"{filtered_df['availability_365'].mean():.0f}")  # средняя доступность

    with st.expander("Описание данных"):  # раскрывающийся блок с таблицей данных
        st.write(filtered_df.head())  # выводим первые строки отфильтрованного DataFrame

    st.subheader("Ценовое распределение")  # заголовок раздела распределения цен
    fig_price = px.histogram(  # строим гистограмму цен
        filtered_df,  # данные для графика
        x="price",  # ось X — цена
        nbins=40,  # количество корзин для гистограммы
        title="Распределение цен Airbnb",  # заголовок графика
        labels={"price": "Цена ($)"},  # подпись оси
    )
    st.plotly_chart(fig_price, use_container_width=True)  # отображаем график Plotly

    st.subheader("Средняя цена по району")  # заголовок раздела средней цены по району
    borough_price = (  # вычисляем среднюю цену по каждому району
        filtered_df.groupby("neighbourhood_group")["price"]
        .mean()
        .reset_index()
        .sort_values(by="price", ascending=False)
    )
    fig_borough = px.bar(  # строим столбчатую диаграмму средней цены по районам
        borough_price,  # данные для графика
        x="neighbourhood_group",  # ось X — район
        y="price",  # ось Y — средняя цена
        title="Средняя цена по району",  # заголовок графика
        labels={"price": "Средняя цена ($)", "neighbourhood_group": "Район"},  # подписи осей
    )
    st.plotly_chart(fig_borough, use_container_width=True)  # отображаем бар-чарт

    st.subheader("Карта объектов")  # заголовок раздела карты
    fig_map = px.scatter_mapbox(  # строим карту точек на Mapbox
        filtered_df,  # данные для карты
        lat="latitude",  # широта
        lon="longitude",  # долгота
        color="room_type",  # цветовая кодировка типа жилья
        size="price",  # размер точки соответствует цене
        hover_name="neighbourhood",  # подсказка при наведении — район
        hover_data={"price": True, "number_of_reviews": True},  # дополнительные поля в подсказке
        zoom=9,  # масштаб карты
        height=450,  # высота графика
        mapbox_style="open-street-map",  # стиль карты
    )
    st.plotly_chart(fig_map, use_container_width=True)  # отображаем карту

    st.subheader("Связь цены и отзывов")  # заголовок раздела зависимости цены и отзывов
    fig_scatter = px.scatter(  # строим scatter-график
        filtered_df,  # данные для графика
        x="number_of_reviews",  # ось X — число отзывов
        y="price",  # ось Y — цена
        color="room_type",  # цвет точек по типу жилья
        title="Зависимость цены от числа отзывов",  # заголовок графика
        labels={"number_of_reviews": "Число отзывов", "price": "Цена ($)"},  # подписи осей
        opacity=0.7,  # прозрачность точек
    )
    st.plotly_chart(fig_scatter, use_container_width=True)  # отображаем scatter-график

    st.subheader("Boxplot цены по типу жилья")  # заголовок раздела boxplot
    fig_box = px.box(  # строим boxplot для цен по типам жилья
        filtered_df,  # данные для графика
        x="room_type",  # ось X — тип жилья
        y="price",  # ось Y — цена
        title="Boxplot цены по типу жилья",  # заголовок графика
    )
    st.plotly_chart(fig_box, use_container_width=True)  # отображаем boxplot

    st.subheader("Корреляционная матрица")  # заголовок тепловой карты корреляций
    corr = filtered_df.select_dtypes(include=["int64", "float64"]).corr()  # вычисляем корреляцию числовых признаков
    fig_corr, ax = plt.subplots(figsize=(10, 6))  # создаём фигуру Matplotlib для тепловой карты
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)  # строим тепловую карту корреляций
    st.pyplot(fig_corr)  # отображаем изображение Matplotlib в Streamlit

    st.markdown("---")  # разделительная линия
    st.markdown("### Основные аналитические выводы")  # заголовок для выводов
    st.markdown(  # текст аналитических выводов
        "- Манхэттен имеет самые высокие средние цены.\n"
        "- Тип жилья сильно влияет на цену: entire home/apt дороже private room.\n"
        "- Высокое количество отзывов чаще встречается у доступных и популярных объектов."
    )


if __name__ == "__main__":
    main()  # запускаем основную функцию при выполнении скрипта напрямую
