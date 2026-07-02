# New York City Airbnb Open Data Analysis

## Описание проекта
Проект посвящён анализу объявлений Airbnb в Нью-Йорке. В рамках работы выполнены очистка данных, EDA, математико-статистический анализ, визуализация результатов, построение простой ML-модели для прогнозирования цены аренды и публикация аналитического продукта в виде интерактивного дашборда.

## Цель проекта
Выявить факторы, влияющие на цену аренды Airbnb в Нью-Йорке, и представить результаты в понятном и воспроизводимом виде.

## Датасет
- Название датасета: New York City Airbnb Open Data
- Источник: Kaggle
- Файл: data/raw/AB_NYC_2019.csv
- Количество строк: 48 895
- Количество признаков: 16
- Основные признаки: price, neighbourhood_group, room_type, minimum_nights, number_of_reviews, reviews_per_month, availability_365, latitude, longitude

## Исследовательские вопросы
1. Какой район Нью-Йорка имеет самые высокие средние цены аренды?
2. Как тип жилья влияет на стоимость объявления?
3. Какие признаки сильнее всего связаны с ценой аренды?
4. Насколько хорошо простая линейная регрессия предсказывает цену?

## Этапы выполнения
1. Загрузка данных
2. Первичный анализ данных / EDA
3. Очистка данных
4. Математико-статистический анализ
5. Визуализация данных
6. Построение интерактивного дашборда
7. Построение ML-модели
8. Интерпретация результатов

## Структура проекта
- data/raw/ — исходный датасет
- data/processed/ — очищенный датасет и результаты анализа
- notebooks/ — ноутбуки с EDA, математическим анализом и ML-моделью
- src/ — скрипты для очистки, анализа, визуализации и обучения модели
- app/ — интерактивный Streamlit-дашборд
- img/ — сохранённые графики и визуализации
- report/ — итоговый отчёт

## Используемые технологии
Python, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, streamlit

## Основные визуализации
- [img/price_distribution.png](img/price_distribution.png)
- [img/borough_price.png](img/borough_price.png)
- [img/room_type_boxplot.png](img/room_type_boxplot.png)
- [img/scatter_reviews_price.png](img/scatter_reviews_price.png)
- [img/correlation_matrix.png](img/correlation_matrix.png)

## ML-модель
- Тип задачи: регрессия
- Использованная модель: Linear Regression
- Целевая переменная: price
- Использованные признаки: minimum_nights, number_of_reviews, reviews_per_month, availability_365, calculated_host_listings_count, neighbourhood_group, room_type
- Метрики качества: MAE, MSE, RMSE, R²

## Результаты
- Выявлены районы и типы жилья с самыми высокими ценами.
- Проведён математико-статистический анализ корреляций и выбросов.
- Построена простая регрессионная модель для предсказания цены аренды.
- Реализован интерактивный дашборд для навигации по данным и результатам анализа.

## Как запустить проект
```bash
pip install -r requirements.txt
python src/data_cleaning.py
python src/math_analysis.py
python src/model.py
python src/visualization.py
streamlit run app/streamlit_app.py
```

## Автор
Аманхан Алихан

