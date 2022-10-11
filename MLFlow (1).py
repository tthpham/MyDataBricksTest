# Databricks notebook source
from mlflow.tracking import MlflowClient
import mlflow

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

my_exp_id = 'd184ab4dccf84ff099dbffb37e9c4ef0'
my_run_id = 'b6029ada783b477c99151411187a5dab'

# COMMAND ----------

type(client.get_run(my_run_id))

# COMMAND ----------

# mlflow.register_model("dbfs:/databricks/mlflow-tracking/d184ab4dccf84ff099dbffb37e9c4ef0/b6029ada783b477c99151411187a5dab/artifacts/model", "model-by-api")

# COMMAND ----------


