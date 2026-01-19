#!/bin/sh
set -e

awslocal s3 mb s3://airflow-xcom || true
awslocal s3 mb s3://nyc-taxi-raw || true
awslocal s3 mb s3://nyc-taxi-processed || true
awslocal s3 mb s3://ml-models || true
