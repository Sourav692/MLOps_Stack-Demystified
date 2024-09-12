from pyspark.sql.functions import struct, lit, to_timestamp
import mlflow.pyfunc

def predict_batch(spark_session, model_uri, input_table_name, output_table_name, model_version, ts):
    """
    Apply the model at the specified URI for batch inference on the table with name input_table_name,
    writing results to the table with name output_table_name
    """
    
    table = spark_session.table(input_table_name)
    
    model = mlflow.pyfunc.spark_udf(spark_session, model_uri=model_uri)
    prediction_df = table.withColumn("prediction", model(struct([table[col] for col in table.columns])))
    
    output_df = (
        prediction_df
        .withColumn("model_version", lit(model_version))
        .withColumn("inference_timestamp", to_timestamp(lit(ts)))
    )
    display(output_df)

    output_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)