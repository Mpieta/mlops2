import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonVirtualenvOperator
from airflow.sdk import Variable
from dotenv import load_dotenv

load_dotenv()

def get_data(data_interval_start, api_key) -> dict:
    from twelvedata import TDClient
        
    td = TDClient(apikey=api_key)
    ts = td.exchange_rate(symbol="USD/EUR", date=data_interval_start)
    return ts.as_json()


def save_data(data: dict) -> None:
    if not data:
        raise ValueError("No data received")

    from airflow.providers.postgres.hooks.postgres import PostgresHook

    POSTGRES_CONN_ID = "postgres_storage"
    hook = PostgresHook.get_hook(POSTGRES_CONN_ID)

    symbol = data["symbol"]
    rate = float(data["rate"])

    hook.run(
        "INSERT INTO exchange_rates (symbol, rate) VALUES (%s, %s)",
        parameters=(symbol, rate),
    )


with DAG(
    dag_id="connections_and_variables",
    start_date=datetime.datetime(2025, 1, 1),
    schedule=datetime.timedelta(minutes=1),
    catchup=True,
) as dag:

    get_data_op = PythonVirtualenvOperator(
        task_id="get_data",
        python_callable=get_data,
        requirements=["twelvedata", "requests"],
        system_site_packages=False,
        python_version="3.11",
        op_kwargs={
            "data_interval_start": "{{ data_interval_start }}",
            "api_key": Variable.get("TWELVEDATA_API_KEY"),
        },
    )

    save_data_op = PythonVirtualenvOperator(
        task_id="save_data",
        python_callable=save_data,
        requirements=["apache-airflow-providers-postgres"],
        system_site_packages=True,
        python_version="3.11",
        op_kwargs={"data": get_data_op.output},
    )

    get_data_op >> save_data_op
