from datetime import datetime, timedelta
from airflow.decorators import dag
from airflow.providers.standard.operators.python import PythonVirtualenvOperator

S3_ENDPOINT = "http://localstack:4566"
S3_PROCESSED_BUCKET = "nyc-taxi-processed"
S3_MODELS_BUCKET = "ml-models"
DATA_YEAR = 2025
POSTGRES_CONN_ID = "postgres_ml"


def load_and_combine_data():
    import polars as pl
    import boto3
    from io import BytesIO
    S3_ENDPOINT = "http://localstack:4566"
    S3_PROCESSED_BUCKET = "nyc-taxi-processed"
    S3_MODELS_BUCKET = "ml-models"
    DATA_YEAR = 2025
    POSTGRES_CONN_ID = "postgres_ml"

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localstack:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

    response = s3_client.list_objects_v2(
        Bucket='nyc-taxi-processed',
        Prefix=f"processed/year={DATA_YEAR}/"
    )

    if 'Contents' not in response:
        raise ValueError("No processed data S3")

    dfs = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.parquet'):
            response = s3_client.get_object(Bucket='nyc-taxi-processed', Key=obj['Key'])
            data = response['Body'].read()
            df = pl.read_parquet(BytesIO(data))
            dfs.append(df)

    df_combined = pl.concat(dfs).sort("date")

    output_buffer = BytesIO()
    df_combined.write_parquet(output_buffer)
    output_buffer.seek(0)

    s3_client.put_object(
        Bucket='ml-models',
        Key='temp/combined_data.parquet',
        Body=output_buffer.getvalue()
    )

    return {
        'data_shape': df_combined.shape,
        'date_range': {
            'min': str(df_combined['date'].min()),
            'max': str(df_combined['date'].max())
        }
    }


def prepare_train_test_split(metadata: dict):
    import polars as pl
    import boto3
    import pickle
    from io import BytesIO
    S3_ENDPOINT = "http://localstack:4566"
    S3_PROCESSED_BUCKET = "nyc-taxi-processed"
    S3_MODELS_BUCKET = "ml-models"
    DATA_YEAR = 2025
    POSTGRES_CONN_ID = "postgres_ml"

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localstack:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

    response = s3_client.get_object(Bucket='ml-models', Key='temp/combined_data.parquet')
    data = response['Body'].read()
    df = pl.read_parquet(BytesIO(data))

    max_month = df['month'].max()

    df_train = df.filter(pl.col('month') < max_month)
    df_test = df.filter(pl.col('month') == max_month)

    feature_cols = [
        'quarter', 'month', 'day_of_month', 'day_of_week', 'is_weekend',
        'airport_rides', 'rush_hour_rides', 'median_trip_dist',
        'avg_fare', 'sum_total_amount', 'sum_congestion_surcharge',
        'total_passengers'
    ]

    X_train = df_train.select(feature_cols).to_numpy()
    y_train = df_train.select('total_rides').to_numpy().ravel()

    X_test = df_test.select(feature_cols).to_numpy()
    y_test = df_test.select('total_rides').to_numpy().ravel()

    train_data = {'X': X_train, 'y': y_train}
    test_data = {'X': X_test, 'y': y_test}

    for name, data in [('train', train_data), ('test', test_data)]:
        buffer = BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)
        s3_client.put_object(
            Bucket='ml-models',
            Key=f'temp/{name}_data.pkl',
            Body=buffer.getvalue()
        )

    return {'train_size': len(X_train), 'test_size': len(X_test)}


def train_ridge_model():
    import boto3
    import pickle
    from io import BytesIO
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error
    S3_ENDPOINT = "http://localstack:4566"
    S3_PROCESSED_BUCKET = "nyc-taxi-processed"
    S3_MODELS_BUCKET = "ml-models"
    DATA_YEAR = 2025
    POSTGRES_CONN_ID = "postgres_ml"

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localstack:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

    response = s3_client.get_object(Bucket='ml-models', Key='temp/train_data.pkl')
    train_data = pickle.load(BytesIO(response['Body'].read()))
    X_train, y_train = train_data['X'], train_data['y']

    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
    }

    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    response = s3_client.get_object(Bucket='ml-models', Key='temp/test_data.pkl')
    test_data = pickle.load(BytesIO(response['Body'].read()))
    X_test, y_test = test_data['X'], test_data['y']

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    model_data = {
        'model': best_model,
        'mae': mae,
        'params': best_params,
        'model_name': 'Ridge'
    }

    buffer = BytesIO()
    pickle.dump(model_data, buffer)
    buffer.seek(0)

    s3_client.put_object(
        Bucket='ml-models',
        Key='temp/ridge_model.pkl',
        Body=buffer.getvalue()
    )

    return {'model_name': 'Ridge', 'mae': float(mae), 's3_key': 'temp/ridge_model.pkl', 'params': best_params}


def train_random_forest_model():
    import boto3
    import pickle
    from io import BytesIO
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    S3_ENDPOINT = "http://localstack:4566"
    S3_PROCESSED_BUCKET = "nyc-taxi-processed"
    S3_MODELS_BUCKET = "ml-models"
    DATA_YEAR = 2025
    POSTGRES_CONN_ID = "postgres_ml"

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localstack:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

    response = s3_client.get_object(Bucket='ml-models', Key='temp/train_data.pkl')
    train_data = pickle.load(BytesIO(response['Body'].read()))
    X_train, y_train = train_data['X'], train_data['y']

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    response = s3_client.get_object(Bucket='ml-models', Key='temp/test_data.pkl')
    test_data = pickle.load(BytesIO(response['Body'].read()))
    X_test, y_test = test_data['X'], test_data['y']

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    model_data = {
        'model': rf,
        'mae': mae,
        'params': {'n_estimators': 100, 'max_depth': 10},
        'model_name': 'RandomForest'
    }

    buffer = BytesIO()
    pickle.dump(model_data, buffer)
    buffer.seek(0)

    s3_client.put_object(
        Bucket='ml-models',
        Key='temp/random_forest_model.pkl',
        Body=buffer.getvalue()
    )

    return {'model_name': 'RandomForest', 'mae': float(mae), 's3_key': 'temp/random_forest_model.pkl',
            'params': {'n_estimators': 100, 'max_depth': 10}}


def train_svm_model():
    import boto3
    import pickle
    import numpy as np
    from io import BytesIO
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error
    S3_ENDPOINT = "http://localstack:4566"
    S3_PROCESSED_BUCKET = "nyc-taxi-processed"
    S3_MODELS_BUCKET = "ml-models"
    DATA_YEAR = 2025
    POSTGRES_CONN_ID = "postgres_ml"

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localstack:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

    response = s3_client.get_object(Bucket='ml-models', Key='temp/train_data.pkl')
    train_data = pickle.load(BytesIO(response['Body'].read()))
    X_train, y_train = train_data['X'], train_data['y']

    subset_size = min(1000, len(X_train))
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]

    svm = SVR(kernel='rbf', C=100, gamma='scale')
    svm.fit(X_train_subset, y_train_subset)

    response = s3_client.get_object(Bucket='ml-models', Key='temp/test_data.pkl')
    test_data = pickle.load(BytesIO(response['Body'].read()))
    X_test, y_test = test_data['X'], test_data['y']

    y_pred = svm.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    model_data = {
        'model': svm,
        'mae': mae,
        'params': {'kernel': 'rbf', 'C': 100, 'training_samples': subset_size},
        'model_name': 'SVM'
    }

    buffer = BytesIO()
    pickle.dump(model_data, buffer)
    buffer.seek(0)

    s3_client.put_object(
        Bucket='ml-models',
        Key='temp/svm_model.pkl',
        Body=buffer.getvalue()
    )

    return {'model_name': 'SVM', 'mae': float(mae), 's3_key': 'temp/svm_model.pkl',
            'params': {'kernel': 'rbf', 'C': 100}}


def select_best_model(ridge_result: dict, rf_result: dict, svm_result: dict):
    import boto3
    from datetime import datetime
    S3_ENDPOINT = "http://localstack:4566"
    S3_PROCESSED_BUCKET = "nyc-taxi-processed"
    S3_MODELS_BUCKET = "ml-models"
    DATA_YEAR = 2025
    POSTGRES_CONN_ID = "postgres_ml"

    models = [ridge_result, rf_result, svm_result]

    best_model = min(models, key=lambda x: x['mae'])

    s3_client = boto3.client(
        's3',
        endpoint_url='http://localstack:4566',
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

    source_key = best_model['s3_key']
    dest_key = f"models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

    s3_client.copy_object(
        Bucket='ml-models',
        CopySource={'Bucket': 'ml-models', 'Key': source_key},
        Key=dest_key
    )

    for model in models:
        s3_client.delete_object(Bucket='ml-models', Key=model['s3_key'])

    s3_client.delete_object(Bucket='ml-models', Key='temp/combined_data.parquet')
    s3_client.delete_object(Bucket='ml-models', Key='temp/train_data.pkl')
    s3_client.delete_object(Bucket='ml-models', Key='temp/test_data.pkl')

    return {
        'best_model': best_model,
        'all_models': models
    }


def log_model_performance(split_info: dict, selection_result: dict):
    import psycopg2
    from datetime import datetime
    S3_ENDPOINT = "http://localstack:4566"
    S3_PROCESSED_BUCKET = "nyc-taxi-processed"
    S3_MODELS_BUCKET = "ml-models"
    DATA_YEAR = 2025
    POSTGRES_CONN_ID = "postgres_ml"

    models = selection_result['all_models']
    train_size = split_info['train_size']

    conn = psycopg2.connect(
        host='postgres-ml',
        port=5432,
        user='postgres',
        password='postgres',
        database='postgres'
    )
    cursor = conn.cursor()

    for model in models:
        cursor.execute(
            """
            INSERT INTO model_performance 
            (training_date, model_name, training_set_size, test_mae, hyperparameters)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            """,
            (
                datetime.now(),
                model['model_name'],
                train_size,
                model['mae'],
                str(model.get('params', {})).replace("'", '"')
            )
        )

    conn.commit()

    cursor.execute(
        """
        SELECT model_name, test_mae, training_set_size 
        FROM model_performance 
        WHERE training_date >= NOW() - INTERVAL '1 hour'
        ORDER BY test_mae
        """
    )

    results = cursor.fetchall()

    cursor.close()
    conn.close()


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


@dag(
    dag_id='nyc_taxi_model_training',
    default_args=default_args,
    description='Train some models',
    schedule='@monthly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
)
def nyc_taxi_model_training_dag():
    load_data = PythonVirtualenvOperator(
        task_id='load_and_combine_data',
        python_callable=load_and_combine_data,
        requirements=['numpy', 'polars>=0.20.3', 'pyarrow>=15.0.0', 'boto3'],
        system_site_packages=False,
        python_version='3.11',
    )
    S3_ENDPOINT = "http://localstack:4566"
    S3_PROCESSED_BUCKET = "nyc-taxi-processed"
    S3_MODELS_BUCKET = "ml-models"
    DATA_YEAR = 2025
    POSTGRES_CONN_ID = "postgres_ml"

    prepare_data = PythonVirtualenvOperator(
        task_id='prepare_train_test_split',
        python_callable=prepare_train_test_split,
        requirements=['numpy','polars>=0.20.3', 'pyarrow>=15.0.0', 'boto3'],
        system_site_packages=False,
        python_version='3.11',
        op_kwargs={'metadata': load_data.output},
    )

    train_ridge = PythonVirtualenvOperator(
        task_id='train_ridge_model',
        python_callable=train_ridge_model,
        requirements=['numpy','scikit-learn>=1.4.0', 'boto3'],
        system_site_packages=False,
        python_version='3.11',
    )

    train_rf = PythonVirtualenvOperator(
        task_id='train_random_forest_model',
        python_callable=train_random_forest_model,
        requirements=['numpy','scikit-learn>=1.4.0', 'boto3'],
        system_site_packages=False,
        python_version='3.11',
    )

    train_svm = PythonVirtualenvOperator(
        task_id='train_svm_model',
        python_callable=train_svm_model,
        requirements=['numpy','scikit-learn>=1.4.0', 'boto3', 'numpy'],
        system_site_packages=False,
        python_version='3.11',
    )

    select_best = PythonVirtualenvOperator(
        task_id='select_best_model',
        python_callable=select_best_model,
        requirements=['numpy','boto3'],
        system_site_packages=False,
        python_version='3.11',
        op_kwargs={
            'ridge_result': train_ridge.output,
            'rf_result': train_rf.output,
            'svm_result': train_svm.output,
        },
    )

    log_performance = PythonVirtualenvOperator(
        task_id='log_model_performance',
        python_callable=log_model_performance,
        requirements=['numpy','psycopg2-binary'],
        system_site_packages=False,
        python_version='3.11',
        op_kwargs={
            'split_info': prepare_data.output,
            'selection_result': select_best.output,
        },
    )
    load_data >> prepare_data >> [train_ridge, train_rf, train_svm] >> select_best >> log_performance


nyc_taxi_model_training_dag()