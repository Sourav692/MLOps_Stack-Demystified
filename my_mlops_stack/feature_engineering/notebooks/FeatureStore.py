# Databricks notebook source

# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Path to the input dataset.
dbutils.widgets.text(
    "datasets_path",
    "dbfs:/mnt/dbacademy-datasets/ml-in-production/v01/airbnb/sf-listings/",
    label="Input Dataset Path",
)

# Input file name.
dbutils.widgets.text(
    "input_file_name",
    "sf-listings.csv",
    label="Input File Name",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "output_table_name",
    "ml_production.airbnb",
    label="Output Feature Table Name",
)

# Primary Keys columns for the feature table;
dbutils.widgets.text(
    "primary_keys",
    "id",
    label="Primary keys columns for the feature table, comma separated.",
)

# Primary Keys columns for the feature table;
dbutils.widgets.text(
    "partition_columns",
    "neighbourhood",
    label="Primary keys columns for the feature table, comma separated.",
)

# COMMAND ----------

# import os
# notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
# %cd $notebook_path
# %cd ../features

# COMMAND ----------

# DBTITLE 1,Define input and output variables
datasets_path = dbutils.widgets.get("datasets_path")
input_file_name = dbutils.widgets.get("input_file_name")
input_table_path = datasets_path + "/"+ input_file_name
output_table_name = dbutils.widgets.get("output_table_name")
pk_columns = dbutils.widgets.get("primary_keys")

assert input_table_path != "", "input_table_path notebook parameter must be specified"
assert output_table_name != "", "output_table_name notebook parameter must be specified"

# Extract database name. Needs to be updated for Unity Catalog.
output_database = output_table_name.split(".")[0]

# COMMAND ----------

# DBTITLE 1,Create database.
spark.sql("CREATE DATABASE IF NOT EXISTS " + output_database)

# COMMAND ----------

# DBTITLE 1, Read input data.
features_df = spark.read.csv(input_table_path, header="true", inferSchema="true", multiLine="true", escape='"')
# COMMAND ----------

# DBTITLE 1, Write computed features.
from databricks import feature_store

fs = feature_store.FeatureStoreClient()

# Create the feature table if it does not exist first.
# Note that this is a no-op if a table with the same name and schema already exists.
fs.create_table(
    name=output_table_name,
    primary_keys=[x.strip() for x in pk_columns.split(",")],
    df=features_df,
    partition_columns = [x.strip() for x in dbutils.widgets.get("partition_columns").split(",")],
)

# Write the computed features dataframe.
fs.write_table(
    name=output_table_name,
    df=features_df,
    mode="merge",
)

# COMMAND ----------

# # DBTITLE 1,Let's reduce the number of review columns by creating an average review score for each listing.
# from pyspark.sql.functions import lit, expr

# review_columns = ["review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
#                   "review_scores_communication", "review_scores_location", "review_scores_value"]

# features_df_short_reviews = (features_df
#                            .withColumn("average_review_score", expr("+".join(review_columns)) / lit(len(review_columns)))
#                            .drop(*review_columns)
#                           )

# fs.write_table(name=output_table_name,
#                df=features_df_short_reviews,
#                mode="overwrite")


# COMMAND ----------

dbutils.notebook.exit(0)