# Databricks notebook source
##################################################################################
# Model Validation Notebook using MLflow
#
# This notebook demonstrates a model validation pipeline using the MLflow model validation API.
# The notebook validates a trained model after it's registered in the Model Registry, ensuring the model 
# meets the required metrics before being deployed to the "Production" stage.
#
# Parameters:
# * env (required)                 - Environment the notebook is run in (staging, or prod). Defaults to "prod".
# * run_mode (required)            - Defines whether model validation is enabled. It can be:
#                                      * 'disabled' - Skips validation.
#                                      * 'dry_run'  - Runs validation without blocking deployment on failures.
#                                      * 'enabled'  - Runs validation and blocks deployment if validation fails.
# * enable_baseline_comparison     - Whether to load the current Production model as a baseline for comparison.
# * model_type (required)          - Type of model ("regressor" or "classifier").
# * targets (required)             - The name of the target column.
# * model_name (required)          - Name of the MLflow registered model.
# * model_version (optional)       - Version of the candidate model to validate.
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import mlflow
import os
import tempfile
import traceback
from mlflow.tracking.client import MlflowClient
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.datasets import fetch_california_housing
from mlflow.models import MetricThreshold

# COMMAND ----------

# DBTITLE 1, Set Notebook Arguments
dbutils.widgets.text("experiment_name", "/dev-my_mlops_stack-experiment-poc", "Experiment Name")
dbutils.widgets.dropdown("run_mode", "dry_run", ["disabled", "dry_run", "enabled"], "Run Mode")
dbutils.widgets.dropdown("enable_baseline_comparison", "false", ["true", "false"], "Enable Baseline Comparison")
dbutils.widgets.text("model_type", "regressor", "Model Type")
dbutils.widgets.text("targets", "MedHouseValue", "Targets")
dbutils.widgets.text("model_name", "RandomForestRegressor_Model", "Model Name")
dbutils.widgets.text("model_version", "1", "Candidate Model Version")

# COMMAND ----------

# DBTITLE 1, Configure Model Validation Settings
# Set the run mode
run_mode = dbutils.widgets.get("run_mode").lower()
dry_run = run_mode == "dry_run"

if run_mode == "disabled":
    print("Model validation is DISABLED. Exiting...")
    dbutils.notebook.exit(0)

if dry_run:
    print(
        "Model validation is in DRY_RUN mode. Validation threshold validation failures will not block model deployment."
    )
else:
    print(
        "Model validation is in ENABLED mode. Validation threshold validation failures will block model deployment."
    )

# Set the experiment
experiment_name = dbutils.widgets.get("experiment_name")
mlflow.set_experiment(experiment_name)

# Set model details
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
model_uri = f"models:/{model_name}/{model_version}" if model_version else None

# Enable baseline comparison
enable_baseline_comparison = dbutils.widgets.get("enable_baseline_comparison") == "true"
baseline_model_uri = f"models:/{model_name}/Production" if enable_baseline_comparison else None

# Set other parameters
model_type = dbutils.widgets.get("model_type")
targets = dbutils.widgets.get("targets")

# COMMAND ----------

# DBTITLE 1, Load Validation Data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df[targets] = data.target

# Splitting data into training and testing sets
X = df.drop(targets, axis=1)
y = df[targets]

# COMMAND ----------

# DBTITLE 1, Define Helper Functions
client = MlflowClient()

def get_run_link(run_info):
    return f"[Run](#mlflow/experiments/{run_info.experiment_id}/runs/{run_info.run_id})"

def log_to_model_description(run, success, model_name, model_version):
    run_link = get_run_link(run.info)
    description = client.get_model_version(model_name, model_version).description
    status = "SUCCESS" if success else "FAILURE"
    new_description = f"{description}\n\nModel Validation Status: {status}\nValidation Details: {run_link}"
    client.update_model_version(model_name, model_version, description=new_description)

def get_training_run(model_name, model_version):
    version = client.get_model_version(model_name, model_version)
    return mlflow.get_run(run_id=version.run_id)


def generate_run_name(training_run):
    return None if not training_run else training_run.info.run_name + "-validation"

def generate_description(training_run):
    return (
        None
        if not training_run
        else "Model Training Details: {0}\n".format(get_run_link(training_run.info))
    )

# COMMAND ----------

# DBTITLE 1, Run Model Evaluation
validation_thresholds = {
    "mean_squared_error": MetricThreshold(0.5, greater_is_better=False),
    "mean_absolute_error": MetricThreshold(0.5, greater_is_better=False),
    "r2_score": MetricThreshold(0.8, greater_is_better=True),
}

training_run = get_training_run(model_name, model_version)

# Run validation within an MLflow run
with mlflow.start_run(run_name="Model_Validation") as run, tempfile.TemporaryDirectory() as tmp_dir:
    validation_thresholds_file = os.path.join(tmp_dir, "validation_thresholds.txt")
    
    # Log validation thresholds to MLflow
    with open(validation_thresholds_file, "w") as f:
        if validation_thresholds:
            for metric_name in validation_thresholds:
                f.write(
                    "{0:30}  {1}\n".format(
                        metric_name, str(validation_thresholds[metric_name])
                    )
                )
    mlflow.log_artifact(validation_thresholds_file)

    # Load the model for evaluation
    model = mlflow.sklearn.load_model(model_uri) 
    predictions = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    # Log metrics
    mlflow.log_metrics({
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2,
    })
    
    # Perform evaluation using mlflow.evaluate
    try:
        # Combine X and y into a single DataFrame
        df_eval = X.copy()
        df_eval[targets] = y

        mlflow.evaluate(
                    model=model_uri,
                    data=df_eval,
                    targets=targets,
                    model_type=model_type,
                    validation_thresholds=validation_thresholds,
                    baseline_model=baseline_model_uri if enable_baseline_comparison else None,
                )

        # Log validation success
        log_to_model_description(run, True, model_name, model_version)
    except Exception as e:
        # Log validation failure
        log_to_model_description(run, False, model_name, model_version)
        
        error_file = os.path.join(tmp_dir, "validation_error.txt")
        with open(error_file, "w") as f:
            f.write(f"Validation failed: {e}\n")
            f.write(traceback.format_exc())
        mlflow.log_artifact(error_file)
        
        if not dry_run:
            raise e

# COMMAND ----------

# DBTITLE 1, Model Validation Completed
print(f"Model validation completed for {model_name}, version {model_version}")