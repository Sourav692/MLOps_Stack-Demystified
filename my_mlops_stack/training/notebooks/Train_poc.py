# Databricks notebook source
##################################################################################
# Model Training Notebook using RandomForest and MLflow
#
# This notebook demonstrates a training pipeline using the RandomForest algorithm.
# It is configured and can be executed as a model training task in an MLOps workflow.
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - MLflow registered model name to use for the trained model.
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# Notebook arguments (provided via widgets or notebook execution arguments)

dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-mlops-experiment",
    label="MLflow experiment name",
)

# MLflow registered model name to use for the trained model.
dbutils.widgets.text(
    "model_name", "RandomForestRegressor_Model", label="Model Name"
)

# COMMAND ----------

# DBTITLE 1, Define input variables
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1, Sample data creation (replace with actual data)
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split

# Fetch California housing data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseValue'] = data.target

# Split the data into training and testing sets
X = df.drop("MedHouseValue", axis=1)
y = df["MedHouseValue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# DBTITLE 1, Train and log the RandomForest model using MLflow
from sklearn.ensemble import RandomForestRegressor
import mlflow.sklearn

# Model parameters
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

# Start MLflow run
run_name = "RandomForestRegressor_Training"
with mlflow.start_run(run_name=run_name) as run:
    
    # Train the model
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="random_forest_model"
    )
    
    # Log the model parameters
    mlflow.log_params(params)

    # Register the model with MLflow
    model_uri = f"runs:/{run.info.run_id}/random_forest_model"
    model_version = mlflow.register_model(model_uri, model_name)

    print(f"Model registered as version {model_version.version} with run ID: {run.info.run_id}")

# COMMAND ----------

# DBTITLE 1, Store model URI and exit
# Store the model URI and version for later retrieval in deployment tasks
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version.version)

# Exit the notebook and pass the model URI as the output
dbutils.notebook.exit(model_uri)