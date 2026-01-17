from datetime import datetime

import pandas as pd
import requests
from airflow.decorators import dag, task


@dag(
    dag_id="weather_data_taskflow_api",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
)
def weather_pipeline():
    @task
    def get_data() -> dict:
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            "?latitude=40.7143&longitude=-74.006"
            "&start_date=2025-01-01&end_date=2025-12-31"
            "&hourly=temperature_2m&timezone=auto"
        )

        resp = requests.get(url)
        resp.raise_for_status()

        data = resp.json()
        return {
            "time": data["hourly"]["time"],
            "temperature": data["hourly"]["temperature_2m"],
        }

    @task
    def transform(data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df["temperature"] = df["temperature"].clip(lower=-20, upper=50)
        return df

    @task
    def save_data(df: pd.DataFrame) -> None:
        df.to_csv("data.csv", index=False)

    data = get_data()
    df = transform(data)
    save_data(df)


weather_pipeline()
