from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "airbnb_clean.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_analysis_tables():
    stats = pd.read_csv(BASE_DIR / "data" / "processed" / "descriptive_statistics.csv")
    corr = pd.read_csv(BASE_DIR / "data" / "processed" / "correlation_matrix.csv")
    price_corr = pd.read_csv(BASE_DIR / "data" / "processed" / "price_correlation.csv")
    metrics = pd.read_csv(BASE_DIR / "data" / "processed" / "model_metrics.csv")
    coefficients = pd.read_csv(BASE_DIR / "data" / "processed" / "model_coefficients.csv")
    return stats, corr, price_corr, metrics, coefficients


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
        (df["neighbourhood_group"].isin(boroughs))
        & (df["room_type"].isin(room_types))
        & (df["price"] >= price_range[0])
        & (df["price"] <= price_range[1])
    ]
    return filtered


def main():
    st.set_page_config(page_title="NYC Airbnb Dashboard", layout="wide", initial_sidebar_state="expanded")

    st.title("NYC Airbnb Open Data Dashboard")
    st.markdown("Анализ структуры, цен и отзывов Airbnb в Нью-Йорке с интерактивными фильтрами, математическим анализом и ML-моделью.")

    df = load_data()
    filtered_df = filter_data(df)
    stats_df, corr_df, price_corr_df, metrics_df, coeff_df = load_analysis_tables()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Объявлений", len(filtered_df))
    col2.metric("Медианная цена", f"${filtered_df['price'].median():.0f}")
    col3.metric("Среднее число отзывов", f"{filtered_df['number_of_reviews'].mean():.1f}")
    col4.metric("Среднее количество дней доступно", f"{filtered_df['availability_365'].mean():.0f}")

    with st.expander("Описание данных"):
        st.dataframe(filtered_df.head(), width="stretch")

    st.subheader("Ценовое распределение")
    fig_price = px.histogram(filtered_df, x="price", nbins=40, title="Распределение цен Airbnb", labels={"price": "Цена ($)"})
    st.plotly_chart(fig_price, width="stretch")

    st.subheader("Средняя цена по району")
    borough_price = (
        filtered_df.groupby("neighbourhood_group")["price"]
        .mean()
        .reset_index()
        .sort_values(by="price", ascending=False)
    )
    fig_borough = px.bar(borough_price, x="neighbourhood_group", y="price", title="Средняя цена по району", labels={"price": "Средняя цена ($)", "neighbourhood_group": "Район"})
    st.plotly_chart(fig_borough, width="stretch")

    st.subheader("Карта объектов")
    fig_map = px.scatter_map(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="room_type",
        size="price",
        hover_name="neighbourhood",
        hover_data={"price": True, "number_of_reviews": True},
        zoom=9,
        height=450,
        map_style="open-street-map",
    )
    st.plotly_chart(fig_map, width="stretch")

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
    st.plotly_chart(fig_scatter, width="stretch")

    st.subheader("Boxplot цены по типу жилья")
    fig_box = px.box(filtered_df, x="room_type", y="price", title="Boxplot цены по типу жилья")
    st.plotly_chart(fig_box, width="stretch")

    st.subheader("Корреляционная матрица")
    corr = filtered_df.select_dtypes(include=["number"]).corr()
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    st.markdown("---")
    st.header("Machine Learning Model and Mathematical Analysis")
    st.markdown("Задача: предсказать цену аренды на основе района, типа жилья, отзывов и минимального числа ночей.")
    st.markdown("- Целевая переменная: `price`")
    st.markdown("- Модель: `LinearRegression`")
    st.markdown("- Признаки: `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `availability_365`, `calculated_host_listings_count`, `neighbourhood_group`, `room_type`")

    st.subheader("Описательная статистика")
    st.dataframe(stats_df.round(3), width="stretch")

    st.subheader("Топ-корреляции с целевой переменной")
    st.dataframe(price_corr_df.head(10).reset_index().rename(columns={"index": "feature"}), width="stretch")

    st.subheader("Метрики качества модели")
    st.dataframe(metrics_df.round(4), width="stretch")

    st.subheader("Коэффициенты модели")
    st.dataframe(coeff_df.head(10), width="stretch")

    col_plot_1, col_plot_2 = st.columns(2)
    with col_plot_1:
        st.image(str(BASE_DIR / "img" / "actual_vs_predicted.png"), caption="Actual vs Predicted")
    with col_plot_2:
        st.image(str(BASE_DIR / "img" / "residual_plot.png"), caption="Residual Plot")

    st.subheader("Вывод по ML-анализу")
    st.markdown(
        "- Линейная регрессия даёт базовый прогноз цены, но не учитывает все сложные нелинейные зависимости.\n"
        "- Район и тип жилья оказывают заметное влияние на цену.\n"
        "- Для будущего улучшения модели можно добавить более сложные модели, новые признаки и учёт сезонности."
    )


if __name__ == "__main__":
    main()
