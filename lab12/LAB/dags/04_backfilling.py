import datetime
import os

import pandas as pd
import pendulum
import requests

from airflow import DAG
from airflow.operators.python import PythonOperator


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "weather_forecast_jan_2025.csv")


def get_weekly_forecast(**kwargs) -> pd.DataFrame:
    logical_date: pendulum.DateTime = kwargs["logical_date"]

    start_date = logical_date.date()
    end_date = (logical_date + datetime.timedelta(days=6)).date()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 40.7143,
        "longitude": -74.006,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ["temperature_2m_min", "temperature_2m_max"],
        "timezone": "America/New_York",
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "forecast_date": logical_date.date().isoformat(),
    })

    return df


def save_data(df: pd.DataFrame) -> None:
    write_header = not os.path.exists(OUTPUT_FILE)

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=write_header,
    )


with DAG(
    dag_id="forecast_backfill",
    start_date=pendulum.datetime(2025, 1, 1, tz="America/New_York"),
    schedule=datetime.timedelta(days=7),
    catchup=True,
) as dag:

    get_data_op = PythonOperator(
        task_id="get_weekly_forecast",
        python_callable=get_weekly_forecast,
    )

    save_data_op = PythonOperator(
        task_id="save_data",
        python_callable=save_data,
        op_kwargs={"df": get_data_op.output},
    )

    get_data_op >> save_data_op
