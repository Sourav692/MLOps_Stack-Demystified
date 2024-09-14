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
    "hive_metastore.default.dummy_inference_data",
    label="Path to the realtime data",
)
realtime_raw_data = dbutils.widgets.get("realtime_raw_data")

# Realtime inference model name
dbutils.widgets.text(
    "model_name", "dev-my_mlops_stack-RandomForestRegressor_Model", label="Model Name"
)
model_name = dbutils.widgets.get("model_name")

# Delta table to store the output predictions.
dbutils.widgets.text("output_table_name", f"{model_name}_predictions", label="Output Table Name")
output_table_name = dbutils.widgets.get("output_table_name")

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

realtime_df = spark.read.table(realtime_raw_data).limit(10)

# COMMAND ----------

from databricks.feature_store import FeatureLookup

# COMMAND ----------

from datetime import timedelta, timezone
import math
import mlflow.pyfunc
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

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
# final_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)