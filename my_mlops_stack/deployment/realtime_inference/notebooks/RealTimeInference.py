# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../../../feature_engineering/features

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text(
    "realtime_raw_data",
    "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
    label="Path to the realtime data",
)
realtime_raw_data = dbutils.widgets.get("realtime_raw_data")

# Pickup features table name
dbutils.widgets.text(
    "pickup_features_table",
    "feature_store_taxi_example.dev_my_mlops_stacks_trip_pickup_features",
    label="Pickup Features Table",
)
pickup_features_table = dbutils.widgets.get("pickup_features_table")

# Dropoff features table name
dbutils.widgets.text(
    "dropoff_features_table",
    "feature_store_taxi_example.dev_my_mlops_stacks_trip_dropoff_features",
    label="Dropoff Features Table",
)
dropoff_features_table = dbutils.widgets.get("dropoff_features_table")

# Realtime inference model name
dbutils.widgets.text(
    "model_name", "dev-my-mlops-stacks-model", label="Model Name"
)
model_name = dbutils.widgets.get("model_name")

# Delta table to store the output predictions.
dbutils.widgets.text("output_table_name", f"{model_name}_predictions", label="Output Table Name")
output_table_name = dbutils.widgets.get("output_table_name")

# Input start date.
dbutils.widgets.text("input_start_date", "", label="Input Start Date")
input_start_date = dbutils.widgets.get("input_start_date")

# Input end date.
dbutils.widgets.text("input_end_date", "", label="Input End Date")
input_end_date = dbutils.widgets.get("input_end_date")

# Timestamp column. Will be used to filter input start/end dates.
# This column is also used as a timestamp key of the feature table.
dbutils.widgets.text(
    "timestamp_column", "", label="Timestamp column"
)
ts_column = dbutils.widgets.get("timestamp_column")

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

realtime_raw_data = spark.read.format("delta").load(realtime_raw_data).limit(10).drop("fareamount")

# COMMAND ----------

from importlib import import_module

mod = import_module("dropoff_features")
compute_features_fn = getattr(mod, "compute_features_fn")

dropoffzip_features_df = compute_features_fn(
    input_df=realtime_raw_data,
    timestamp_column="tpep_dropoff_datetime",
    start_date=input_start_date,
    end_date=input_end_date,
)

pickupzip_features_df = compute_features_fn(
    input_df=realtime_raw_data,
    timestamp_column="tpep_pickup_datetime",
    start_date=input_start_date,
    end_date=input_end_date,
)

# COMMAND ----------

from databricks.feature_store import FeatureLookup

pickup_feature_lookups = [
    FeatureLookup(
        table_name=pickup_features_table,
        feature_names=[
            "mean_fare_window_1h_pickup_zip",
            "count_trips_window_1h_pickup_zip",
        ],
        lookup_key=["pickup_zip"],
        timestamp_lookup_key=["rounded_pickup_datetime"],
    ),
]

dropoff_feature_lookups = [
    FeatureLookup(
        table_name=dropoff_features_table,
        feature_names=["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
        lookup_key=["dropoff_zip"],
        timestamp_lookup_key=["rounded_dropoff_datetime"],
    ),
]

# COMMAND ----------

from datetime import timedelta, timezone
import math
import mlflow.pyfunc
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())


def rounded_taxi_data(taxi_data_df):
    # Round the taxi data timestamp to 15 and 30 minute intervals so we can join with the pickup and dropoff features
    # respectively.
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df["tpep_pickup_datetime"], F.lit(15)
                )
            ),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df["tpep_dropoff_datetime"], F.lit(30)
                )
            ),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df

# COMMAND ----------

from databricks import feature_store

realtime_data = rounded_taxi_data(realtime_raw_data)
from databricks import feature_store

# Since the rounded timestamp columns would likely cause the model to overfit the data
# unless additional feature engineering was performed, exclude them to avoid training on them.
exclude_columns = ["rounded_pickup_datetime", "rounded_dropoff_datetime"]

fs = feature_store.FeatureStoreClient()

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
realtime_set = fs.create_training_set(
    realtime_data,
    feature_lookups=pickup_feature_lookups + dropoff_feature_lookups,
    label="fare_amount",
    exclude_columns=exclude_columns,
)

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
realtime_df = realtime_set.load_df().drop("fare_amount")

# COMMAND ----------

realtime_df

# COMMAND ----------

import requests
import numpy as np
import pandas as pd
import json
import ssl

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):  
    url = f'{API_ROOT}/serving-endpoints/{model_name}/invocations'
    headers = {'Authorization': f'Bearer {API_TOKEN}',
'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json, verify = ssl.CERT_NONE)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return data_json, response.json()

# COMMAND ----------

predictions=score_model(realtime_df.toPandas())
json_string = predictions[0]

# COMMAND ----------

# Parse JSON string into a dictionary
data_dict = json.loads(json_string)

# Extract the DataFrame part from the dictionary
dataframe_dict = data_dict['dataframe_split']

# Create DataFrame
df = pd.DataFrame(dataframe_dict['data'], columns=dataframe_dict['columns'], index=dataframe_dict['index'])


# COMMAND ----------

predictions_df = pd.DataFrame(data=predictions[1]['predictions'], columns=['Predictions'])
final_df = pd.concat([df, predictions_df], axis=1)
final_df.display()

# COMMAND ----------