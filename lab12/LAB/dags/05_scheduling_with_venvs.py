import datetime
import json
import os
from airflow import DAG
from airflow.providers.standard.operators.python import PythonVirtualenvOperator
from dotenv import load_dotenv

load_dotenv()



def get_data(data_interval_start) -> dict:
    from twelvedata import TDClient
        
    td = TDClient(apikey=os.environ["TWELVEDATA_API_KEY"])
    ts = td.exchange_rate(symbol="USD/EUR", date=data_interval_start)
    return ts.as_json()


def save_data(data: dict) -> None:
    if not data:
        raise ValueError("No data received")
    
    DATA_FILE = "data.jsonl"
    with open(DATA_FILE, "a+") as file:
        file.write(json.dumps(data))
        file.write("\n")


with DAG(
    dag_id="scheduling_dataset_gathering",
    start_date=datetime.datetime(2025, 1, 1),
    schedule=datetime.timedelta(minutes=1),
    catchup=True,
) as dag:

    get_data_op = PythonVirtualenvOperator(
        task_id="get_data",
        python_callable=get_data,
        requirements=["twelvedata", "pendulum", "lazy_object_proxy", "requests"],
        system_site_packages=False,
        python_version="3.11",
        op_kwargs={"data_interval_start": "{{ data_interval_start }}"},
    )

    save_data_op = PythonVirtualenvOperator(
        task_id="save_data",
        python_callable=save_data,
        requirements=[],
        system_site_packages=True,
        python_version="3.11",
        op_kwargs={"data": get_data_op.output},
    )

    get_data_op >> save_data_op
