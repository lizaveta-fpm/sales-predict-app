from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

# Автоматически определяем абсолютный путь к корню проекта
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Базовые настройки для перезапуска в случае сбоев
default_args = {
    'owner': 'elizaveta',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'retail_sales_forecasting_pipeline',
    default_args=default_args,
    description='Автоматический пересчет прогнозов розничной сети (XGBoost, CatBoost, LSTM)',
    schedule_interval='@weekly',  # Запуск раз в неделю.
    catchup=False,
    max_active_runs=1
) as dag:

    # запуск тяжелых вычислений
    run_ml_pipeline = BashOperator(
        task_id='generate_new_predictions',
        bash_command=f'cd {BASE_DIR} && python generate_predictions.py',
    )