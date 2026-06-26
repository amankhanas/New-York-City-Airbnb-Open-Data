import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/airbnb_clean.csv")


def filter_data(df):
    boroughs = st.sidebar.multiselect(
        "Выберите район (borough)",
        options=df["neighbourhood_group"].unique(),
        default=df["neighbourhood_group"].unique(),
    )
    room_types = st.sidebar.multiselect(
        "Выберите тип жилья",
        options=df["room_type"].unique(),
        default=df["room_type"].unique(),
    )
    price_range = st.sidebar.slider(
        "Диапазон цены",
        int(df["price"].min()),
        int(df["price"].max()),
        (int(df["price"].quantile(0.05)), int(df["price"].quantile(0.95))),
    )

    filtered = df[
        (df["neighbourhood_group"].isin(boroughs)) &
        (df["room_type"].isin(room_types)) &
        (df["price"] >= price_range[0]) &
        (df["price"] <= price_range[1])
    ]
    return filtered


def main():
    st.set_page_config(
        page_title="NYC Airbnb Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("NYC Airbnb Open Data Dashboard")
    st.markdown(
        "Анализ структуры, цен и отзывов Airbnb в Нью-Йорке с интерактивными фильтрами и визуализацией."
    )

    df = load_data()
    filtered_df = filter_data(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Объявлений", len(filtered_df))
    col2.metric("Медианная цена", f"${filtered_df['price'].median():.0f}")
    col3.metric("Среднее число отзывов", f"{filtered_df['number_of_reviews'].mean():.1f}")
    col4.metric("Среднее количество дней доступно", f"{filtered_df['availability_365'].mean():.0f}")

    with st.expander("Описание данных"):
        st.write(filtered_df.head())

    st.subheader("Ценовое распределение")
    fig_price = px.histogram(
        filtered_df,
        x="price",
        nbins=40,
        title="Распределение цен Airbnb",
        labels={"price": "Цена ($)"},
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Средняя цена по району")
    borough_price = (
        filtered_df.groupby("neighbourhood_group")["price"]
        .mean()
        .reset_index()
        .sort_values(by="price", ascending=False)
    )
    fig_borough = px.bar(
        borough_price,
        x="neighbourhood_group",
        y="price",
        title="Средняя цена по району",
        labels={"price": "Средняя цена ($)", "neighbourhood_group": "Район"},
    )
    st.plotly_chart(fig_borough, use_container_width=True)

    st.subheader("Карта объектов")
    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="room_type",
        size="price",
        hover_name="neighbourhood",
        hover_data={"price": True, "number_of_reviews": True},
        zoom=9,
        height=450,
        mapbox_style="open-street-map",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Связь цены и отзывов")
    fig_scatter = px.scatter(
        filtered_df,
        x="number_of_reviews",
        y="price",
        color="room_type",
        title="Зависимость цены от числа отзывов",
        labels={"number_of_reviews": "Число отзывов", "price": "Цена ($)"},
        opacity=0.7,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Boxplot цены по типу жилья")
    fig_box = px.box(
        filtered_df,
       x="room_type",
        y="price",
        title="Boxplot цены по типу жилья",
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Корреляционная матрица")
    corr = filtered_df.select_dtypes(include=["int64", "float64"]).corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    st.markdown("---")
    st.markdown("### Основные аналитические выводы")
    st.markdown(
        "- Манхэттен имеет самые высокие средние цены.\n"
        "- Тип жилья сильно влияет на цену: entire home/apt дороже private room.\n"
        "- Высокое количество отзывов чаще встречается у доступных и популярных объектов."
    )


if __name__ == "__main__":
    main()
