# Databricks notebook source
import databricks.automl as automl
from pyspark.sql import DataFrame

# COMMAND ----------

df = sql('SELECT * FROM default.transactions_train')
display(df)

# COMMAND ----------

rd_seed = 27
target = 'authorized_flag'

# COMMAND ----------

dftrain, dftest = df.randomSplit([.8, .2], seed=rd_seed)

# COMMAND ----------

summary_forecast = automl.forecast(dataset=dftrain, target_col='purchase_amount', max_trials=12, timeout_minutes=5, primary_metric='mae', time_col='purchase_date')

# COMMAND ----------

summary

# COMMAND ----------


