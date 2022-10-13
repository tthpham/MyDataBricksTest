# Databricks notebook source
# DBTITLE 1,ML Workflow
# MAGIC %md
# MAGIC - Exploratory data analysis
# MAGIC - Feature engineering
# MAGIC - Hyperparameter tuning
# MAGIC - Evaluation and selection

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from sparkdl.xgboost import XgboostClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import hyperopt
import mlflow

# COMMAND ----------

random_seed = 27

# COMMAND ----------

fsc = FeatureStoreClient()

# COMMAND ----------

# Get data from Feature Store
dfw = fsc.read_table(name='my_features.wine_data')

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructField, StructType
 
schema = StructType([
  StructField("age", DoubleType(), False),
  StructField("workclass", StringType(), False),
  StructField("fnlwgt", DoubleType(), False),
  StructField("education", StringType(), False),
  StructField("education_num", DoubleType(), False),
  StructField("marital_status", StringType(), False),
  StructField("occupation", StringType(), False),
  StructField("relationship", StringType(), False),
  StructField("race", StringType(), False),
  StructField("sex", StringType(), False),
  StructField("capital_gain", DoubleType(), False),
  StructField("capital_loss", DoubleType(), False),
  StructField("hours_per_week", DoubleType(), False),
  StructField("native_country", StringType(), False),
  StructField("income", StringType(), False)
])
 
df = spark.read.format("csv").schema(schema).load("/databricks-datasets/adult/adult.data")
cols = df.columns

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.describe())

# COMMAND ----------

display(df.summary())

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# Create binary feature for column with missing values
na_cols = ['capital_gain', 'capital_loss']
c = 'capital_gain'
display(df.withColumn(f'{c}_na', when(col(c).isNull(), 1).otherwise(0)).select([c, f'{c}_na']))

# COMMAND ----------

org_target = 'income'
label = 'label'
cat_features = [c for c, data_type in df.dtypes if data_type=='string' and c != org_target]
num_features = [c for c, data_type in df.dtypes if data_type=='double' and c != org_target]
print("Categorical features: ", cat_features)
print("Numerical features: ", num_features)

# COMMAND ----------

string_indexer = StringIndexer(inputCols=cat_features, outputCols=[f'{c}_Index' for c in cat_features])  # convert the values "red", "blue", and "green" to 0, 1, and 2
encoder = OneHotEncoder(inputCols=string_indexer.getOutputCols(), outputCols=[f'{c}_ohe' for c in cat_features])
label_indexer = StringIndexer(inputCol=org_target, outputCol=label)

# COMMAND ----------

string_indexer_model = string_indexer.fit(df)
display(string_indexer_model.transform(df))

# COMMAND ----------

encoder_model = encoder.fit(string_indexer_model.transform(df))
display(encoder_model.transform(string_indexer_model.transform(df)))

# COMMAND ----------

dftrain, dftest = df.randomSplit([.8, .2], seed=random_seed)

# COMMAND ----------

mlfeatures = num_features + string_indexer.getOutputCols()
print('ML features', mlfeatures)
# Combine all feature columns into a single feature vector
vector_assembler = VectorAssembler(inputCols=mlfeatures, outputCol='mlfeatures')
# Estimator
estimator = LogisticRegression(labelCol=label, featuresCol='mlfeatures', regParam=1.0)
# Create a pipeline
pl = Pipeline(stages=[string_indexer, label_indexer, vector_assembler, estimator])

# COMMAND ----------

# Baseline model
with mlflow.start_run(run_name='run-lr-income-prediction') as run:
    mlflow.autolog()
    plmodel = pl.fit(dftrain)
    # Calculate lmetrics
    dfpred = plmodel.transform(dftest)
    evaluator_roc = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol='prediction', metricName="areaUnderROC")
    roc = evaluator.evaluate(dfpred)
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=label, predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator_accuracy.evaluate(dfpred)
    print(f'ROC: {roc}, Accuracy: {accuracy}')

# COMMAND ----------


