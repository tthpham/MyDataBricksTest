# Databricks notebook source
from databricks.feature_store import FeatureStoreClient
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

df = fs.read_table(name="my_features.dataset_airbnb")
# Split data
train_df, test_df = df.randomSplit([.8, .2], seed=27)

# COMMAND ----------

file_path = "dbfs:/mnt/dbacademy-datasets/scalable-machine-learning-with-apache-spark/v02/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

cat_cols = [name for name, dtype in df.dtypes if dtype == 'string']
index_cat_cols = [f'index_{col}' for col in cat_cols]
ml_features = [c for c, dtype in df.dtypes if dtype == 'double' and c not in ['index', 'price']] + index_cat_cols
print(ml_features)

# COMMAND ----------

# String Indexer
str_indexer = StringIndexer(inputCols=cat_cols, outputCols=index_cat_cols, handleInvalid='skip')

# COMMAND ----------

# Vector Assembler
va = VectorAssembler(inputCols=ml_features, outputCol='features', handleInvalid='skip')

# COMMAND ----------

# RF model
rf = RandomForestRegressor(labelCol='price', seed=27, maxBins=40)
# LR
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol='features', labelCol="price")
from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(labelCol="price", maxBins=40)

# COMMAND ----------

# Create a pipeline
pl = Pipeline(stages=[str_indexer, va, dt])

# COMMAND ----------

pl_model = pl.fit(train_df)

# COMMAND ----------

# rf_model = pl_model.stages[-1]

# COMMAND ----------

dt_model = pl_model.stages[-1]
display(dt_model)

# COMMAND ----------


