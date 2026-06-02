import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Настройка страницы
st.set_page_config(page_title="Прогноз Продаж", layout="wide")
st.title("📊 Автоматизированное планирование поставок")

@st.cache_data
def load_predictions_database():
    try:
        df = pd.read_parquet('precomputed_predictions.parquet')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("ошибка: Файл базы данных 'precomputed_predictions.parquet' не найден. Сначала запустите 'generate_predictions.py'.")
        st.stop()

db = load_predictions_database()

st.sidebar.header("Параметры аналитики")

available_horizons = sorted(db['horizon'].unique().tolist())
horizon = st.sidebar.selectbox(
    "Горизонт прогноза (дней)", 
    available_horizons, 
    index=available_horizons.index(16) if 16 in available_horizons else 0,
    help="Выберите период планирования закупки"
)

available_models = sorted(db['model_type'].unique().tolist())
selected_model = st.sidebar.selectbox("Алгоритм прогнозирования", available_models, index=0)

# Фильтрация глобального среза под выбранную конфигурацию
slice_mask = (db['horizon'] == horizon) & (db['model_type'] == selected_model)
global_slice = db.loc[slice_mask]

# Динамическое формирование списков для выбора объектов
stores = sorted(global_slice['store_nbr'].unique().tolist())
families = sorted(global_slice['family'].unique().tolist())

st.sidebar.header("Выбор объекта")
selected_store = st.sidebar.selectbox("Номер магазина", stores, index=0)
selected_family = st.sidebar.selectbox("Категория товара", families, index=0)

global_test = global_slice[global_slice['data_type'] == 'test']
y_g_true = global_test['sales'].values
y_g_pred = global_test['predictions'].values

# Безопасный расчет глобальных метрик с защитой от пустых массивов
if len(y_g_true) > 0:
    global_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_g_true, y_g_pred)),
        'mae': mean_absolute_error(y_g_true, y_g_pred),
        'wape': (np.sum(np.abs(y_g_true - y_g_pred)) / np.sum(y_g_true)) * 100 if np.sum(y_g_true) > 0 else 0.0,
        'r2': r2_score(y_g_true, y_g_pred) if len(y_g_true) > 1 else 0.0
    }
else:
    global_metrics = {'rmse': 0.0, 'mae': 0.0, 'wape': 0.0, 'r2': 0.0}

# Фильтрация данных под локальный объект
object_slice = global_slice[(global_slice['store_nbr'] == selected_store) & (global_slice['family'] == selected_family)]
item_test = object_slice[object_slice['data_type'] == 'test'].sort_values('date')
item_hist = object_slice[object_slice['data_type'] == 'history'].sort_values('date')

# Локальные метрики
y_true_local = item_test['sales'].values
y_pred_local = item_test['predictions'].values

local_rmse = np.sqrt(mean_squared_error(y_true_local, y_pred_local)) if len(y_true_local) > 0 else 0.0
local_mae = mean_absolute_error(y_true_local, y_pred_local) if len(y_true_local) > 0 else 0.0
local_wape = (np.sum(np.abs(y_true_local - y_pred_local)) / np.sum(y_true_local)) * 100 if np.sum(y_true_local) > 0 else 0.0

total_demand = int(np.ceil(item_test['predictions'].sum())) if len(item_test) > 0 else 0

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label=f"🛒 ОБЪЕМ ЗАКУПКИ НА {horizon} ДН.",
        value=f"{total_demand:,} ед."
    )
with col2:
    if len(item_test) > 0:
        st.info(f"📍 Магазин № {selected_store} | Категория: {selected_family}\nИнтервал прогноза: {item_test['date'].min().strftime('%Y-%m-%d')} — {item_test['date'].max().strftime('%Y-%m-%d')}")
    else:
        st.warning("Нет тестовых данных для выбранной комбинации за указанный период.")

st.markdown("### 🎯 Точность прогноза для выбранной комбинации")
loc_col1, loc_col2, loc_col3 = st.columns(3)

loc_col1.metric(
    label="Loc_WAPE (Ошибка)", 
    value=f"{local_wape:.2f} %",
    delta=f"{local_wape - global_metrics['wape']:.2f}% от среднего по сети",
    delta_color="inverse"
)
loc_col2.metric(label="Локальный MAE", value=f"{local_mae:.2f} ед.")
loc_col3.metric(label="Локальный RMSE", value=f"{local_rmse:.2f}")

# График
st.subheader("История продаж и прогноз")

chart_hist = pd.DataFrame({'Исторический спрос': item_hist['sales'].values}, index=item_hist['date'])
chart_fact = pd.DataFrame({'Реальный факт (Тест)': item_test['sales'].values}, index=item_test['date'])
chart_pred = pd.DataFrame({f'Прогноз {selected_model}': item_test['predictions'].values}, index=item_test['date'])

combined_chart = pd.concat([chart_hist, chart_fact, chart_pred], axis=1)
st.line_chart(combined_chart)

# Вкладки с таблицами и глобальным качеством
tab1, tab2 = st.tabs(["Пошаговый план закупки", "Общие метрики точности модели"])

with tab1:
    st.markdown("Таблица суточной потребности:")
    if len(item_test) > 0:
        output_table = item_test[['date', 'predictions']].copy()
        output_table.columns = ['Дата поставки', 'Необходимый объем (штук)']
        st.dataframe(output_table.set_index('Дата поставки'), use_container_width=True)
    else:
        st.caption("Данные для формирования таблицы отсутствуют.")

with tab2:
    st.markdown(f"Показатели качества работы алгоритма **{selected_model}**, рассчитанные по всей розничной сети:")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("WAPE (Ошибка всей сети)", f"{global_metrics['wape']:.2f} %")
    m_col2.metric("MAE (Средняя ошибка)", f"{global_metrics['mae']:.2f} ед.")
    m_col3.metric("RMSE", f"{global_metrics['rmse']:.2f}")
    m_col4.metric("Коэффициент R²", f"{global_metrics['r2']:.4f}")

