from datetime import datetime, timedelta
from airflow.decorators import dag
from airflow.providers.standard.operators.python import PythonVirtualenvOperator

S3_ENDPOINT = "http://localstack:4566"
S3_RAW_BUCKET = "nyc-taxi-raw"
S3_PROCESSED_BUCKET = "nyc-taxi-processed"



def download_taxi_data(logical_date=None, **context):
    import boto3
    import urllib.request
    from datetime import datetime
    DATA_YEAR = 2025

    if isinstance(logical_date, str):
        logical_date = datetime.fromisoformat(logical_date.replace('Z', '+00:00'))

    month = logical_date.month
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{DATA_YEAR}-{month:02}.parquet"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    with urllib.request.urlopen(req) as response:
        if response.status != 200:
            raise ValueError(f"Download failed with status: {response.status}")
        data = response.read()

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localstack:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )
    s3_key = f"raw/year={DATA_YEAR}/month={month:02}/data.parquet"

    s3_client.put_object(
        Bucket='nyc-taxi-raw',
        Key=s3_key,
        Body=data
    )
    return s3_key


def process_single_month(s3_key: str, logical_date=None, **context):
    import polars as pl
    import boto3
    from io import BytesIO
    from datetime import datetime
    DATA_YEAR = 2025
    if isinstance(logical_date, str):
        logical_date = datetime.fromisoformat(logical_date.replace('Z', '+00:00'))

    month = logical_date.month

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localstack:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

    response = s3_client.get_object(Bucket='nyc-taxi-raw', Key=s3_key)
    data = response['Body'].read()

    df = pl.read_parquet(BytesIO(data))

    money_cols = [
        "fare_amount", "extra", "mta_tax", "tip_amount",
        "tolls_amount", "improvement_surcharge", "total_amount",
        "congestion_surcharge", "Airport_fee"
    ]

    df_processed = (
        df
        .with_columns([
            pl.col("tpep_pickup_datetime").cast(pl.Datetime("ms")),
            pl.col("tpep_dropoff_datetime").cast(pl.Datetime("ms"))
        ])
        .with_columns(
            trip_time=(pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_minutes()
        )
        .filter(
            (pl.col("tpep_pickup_datetime").dt.year() == 2025) &
            (pl.col("tpep_dropoff_datetime").dt.date() <= pl.date(2026, 1, 1))
        )
        .with_columns(pl.col("passenger_count").fill_null(1))
        .filter(pl.col("passenger_count") != 0)
        .with_columns(
            passenger_count=pl.when(pl.col("passenger_count") > 6)
            .then(pl.lit(6))
            .otherwise(pl.col("passenger_count"))
        )
        .filter(pl.col("trip_time") <= 120)
        .with_columns([pl.col(col).abs() for col in money_cols])
        .filter(~pl.any_horizontal([pl.col(col) > 1000 for col in money_cols]))
        .filter(pl.col("VendorID").is_not_null() & pl.col("VendorID").is_in([1, 2, 6, 7]))
        .filter(pl.col("RatecodeID").is_not_null() & pl.col("RatecodeID").is_in([1, 2, 3, 4, 5, 6, 99]))
        .with_columns(
            pl.col("payment_type").replace_strict({
                1: "card", 2: "cash",
                3: "other", 4: "other", 5: "other", 6: "other"
            }).cast(pl.Categorical)
        )
        .with_columns([
            (pl.col("Airport_fee") > 0).alias("is_airport_ride"),
            (
                (
                    (pl.col("tpep_pickup_datetime").dt.time().is_between(pl.time(6, 30), pl.time(9, 30))) |
                    (pl.col("tpep_pickup_datetime").dt.time().is_between(pl.time(15, 30), pl.time(20, 0)))
                ) &
                pl.col("tpep_pickup_datetime").dt.weekday().is_in([1, 2, 3, 4, 5])
            ).alias("is_rush_hour")
        ])
    )
    df_daily = (
        df_processed
        .with_columns([
            pl.col("tpep_pickup_datetime").dt.date().alias("date"),
            pl.col("tpep_pickup_datetime").dt.month().alias("month"),
            pl.col("tpep_pickup_datetime").dt.quarter().alias("quarter"),
            pl.col("tpep_pickup_datetime").dt.day().alias("day_of_month"),
            pl.col("tpep_pickup_datetime").dt.weekday().alias("day_of_week"),
            pl.col("tpep_pickup_datetime").dt.weekday().is_in([6, 7]).alias("is_weekend")
        ])
        .group_by(["date", "quarter", "month", "day_of_month", "day_of_week", "is_weekend"])
        .agg([
            pl.len().alias("total_rides"),
            pl.sum("is_airport_ride").alias("airport_rides"),
            pl.sum("is_rush_hour").alias("rush_hour_rides"),
            pl.col("trip_distance").median().alias("median_trip_dist"),
            pl.mean("fare_amount").alias("avg_fare"),
            pl.sum("total_amount").alias("sum_total_amount"),
            pl.sum("congestion_surcharge").alias("sum_congestion_surcharge"),
            pl.sum("passenger_count").alias("total_passengers")
        ])
        .sort("date")
    )

    output_buffer = BytesIO()
    df_daily.write_parquet(output_buffer)
    output_buffer.seek(0)

    output_key = f"processed/year={DATA_YEAR}/month={month:02}/daily_aggregates.parquet"
    s3_client.put_object(
        Bucket='nyc-taxi-processed',
        Key=output_key,
        Body=output_buffer.getvalue()
    )

    return {
        'output_key': output_key,
        'rows': len(df_daily),
        'month': month
    }


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='nyc_taxi_data_processing',
    default_args=default_args,
    description='Download and process taxi data',
    schedule='@monthly',
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31),
    catchup=True,
    max_active_runs=3,
)
def nyc_taxi_data_processing_dag():

    download_task = PythonVirtualenvOperator(
        task_id='download_taxi_data',
        python_callable=download_taxi_data,
        requirements=['boto3'],
        system_site_packages=False,
        python_version='3.11',
        op_kwargs={'logical_date': '{{ logical_date }}'},
    )

    process_task = PythonVirtualenvOperator(
        task_id='process_single_month',
        python_callable=process_single_month,
        requirements=['polars>=0.20.3', 'pyarrow>=15.0.0', 'boto3'],
        system_site_packages=False,
        python_version='3.11',
        op_kwargs={
            's3_key': download_task.output,
            'logical_date': '{{ logical_date }}'
        },
    )

    download_task >> process_task

nyc_taxi_data_processing_dag()