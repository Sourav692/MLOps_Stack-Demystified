# Databricks notebook source
import os
import sys
import requests
import json
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# COMMAND ----------

# Set the name of the MLflow endpoint
endpoint_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="")
 
# Name of the registered MLflow model
model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="")
 
# Get the latest version of the MLflow model
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="")
 
# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "CPU" 
 
# Specify the scale-out size of compute (Small, Medium, Large, etc.)
workload_size = "Small" 
 
# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

serving_endpoint_response= requests.get(
    url=f"{API_ROOT}/api/2.0/serving-endpoints/{endpoint_name}",headers=headers
)

# COMMAND ----------

if serving_endpoint_response.status_code == 200:
    data={
    "served_entities": [
        {
        "name": f"{model_name}-{model_version}",
        "entity_name": model_name,
        "entity_version": model_version,
        "workload_size": workload_size,
        "scale_to_zero_enabled": True
        }
    ]
    }
    response = requests.put(
            url=f"{API_ROOT}/api/2.0/serving-endpoints/{endpoint_name}/config", json=data, headers=headers
        )
    print(json.dumps(response.json(), indent=4))
else:
    data={
    "name": endpoint_name,
    "config": {
        "served_entities": [
        {
            "name": f"{model_name}-{model_version}",
            "entity_name": model_name,
            "entity_version": model_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": True
        }
        ]
    },
    "tags": [
        {
        "key": "team",
        "value": "MLOps"
        }
    ]
    }
 
    response = requests.post(
        url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers
    )
 
    print(json.dumps(response.json(), indent=4))