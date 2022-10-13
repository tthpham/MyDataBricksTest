# Databricks notebook source
from pyspark.sql.functions import lit, col, monotonically_increasing_id
from pyspark.sql import DataFrame
import mlflow
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from sparkdl.xgboost import XgboostClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from databricks.feature_store import FeatureStoreClient
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from numpy.random import default_rng
from pyspark.ml import Pipeline

# COMMAND ----------

random_seed = 27

# COMMAND ----------

dfww = spark.read.format("csv").option("header", "true").option("sep", ";").load("dbfs:/FileStore/shared_uploads/tthpham2710@outlook.com/winequality_white.csv")
dfrw = spark.read.format("csv").option("header", "true").option("sep", ";").load("dbfs:/FileStore/shared_uploads/tthpham2710@outlook.com/winequality_red.csv")

# COMMAND ----------

# Convert string datatype to double datatype
for col_name in [col_name for col_name, col_type in dfrw.dtypes if col_type == 'string']:
    dfrw = dfrw.withColumn(col_name, col(col_name).cast('double'))
    dfww = dfww.withColumn(col_name, col(col_name).cast('double'))

# COMMAND ----------

dfrw = dfrw.withColumn("is_red", lit(1))
dfww = dfww.withColumn("is_red", lit(0))

# COMMAND ----------

# Merge 2 dataframes
df = DataFrame.unionAll(dfrw, dfww)
display(df)

# COMMAND ----------

map_col_names = {c: c.replace(' ', '_') for c in df.columns}
df = df.select([col(c).alias(map_col_names.get(c, c)) for c in df.columns])

# COMMAND ----------

df = df.withColumn('is_good_quality', (col('quality') >= 7).cast('double'))
display(df)

# COMMAND ----------

df = df.withColumn('id', monotonically_increasing_id())

# COMMAND ----------

display(df)

# COMMAND ----------

# Store data into Feature Store
fsc = FeatureStoreClient()
table_name = 'my_features.wine_data'

# COMMAND ----------

# Create table
data_table = fsc.create_table(
    name=table_name,
    df=df,
    primary_keys='id'
)

# COMMAND ----------

# Get data from Feature Store
dfwine = fsc.read_table(name=table_name)

# COMMAND ----------

# split data for train and test
dftrain, dftest = dfwine.randomSplit([.8, .2], seed=random_seed)

# COMMAND ----------

target = 'is_good_quality'
input_features = [c for c in dfwine.columns if c not in [target, 'quality', 'id']]
print(input_features)

# COMMAND ----------

mlflow.autolog()
with mlflow.start_run(run_name='rf_quality') as run:
    # vector assembler
    va = VectorAssembler(inputCols=input_features, outputCol='features')
    estimator = RandomForestClassifier(featuresCol='features', labelCol=target)
    pl = Pipeline(stages=[va, estimator])
    plmodel = pl.fit(dftrain)
    
    # log parameters
    # mlflow.log_params({
    #    'target': target,
    #    'features': input_features,
    #    'pipeline': 'vector assember, random forest classifier'
    # })
    # make prediction
    dfpred = plmodel.transform(dftest)
    display(dfpred)
    
    # calculate metrics
    evaluator = BinaryClassificationEvaluator(labelCol=target, rawPredictionCol='prediction')
    my_metrics = evaluator.evaluate(dfpred)
    
    # register models
    # mlflow.register_model(plmodel, name='registered-rf-wine-classifier')

# COMMAND ----------

# define hyperparameter optimization parameters
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}

# COMMAND ----------

def train_models(params):
    # enable auto logging for each worker
    mlflow.autolog()
    # create run
    with mlflow.start_run():
        va = VectorAssembler(inputCols=input_features, outputCol='features')
        estimator = XgboostClassifier(labelCol=target, random_state=random_seed, **params)
        pl = Pipeline(stages=[va, estimator])
        plmodel = pl.fit(dftrain)
        # prediction
        dfpred = plmodel.transform(dftest)
        # calculate metrics
        evaluator = BinaryClassificationEvaluator(labelCol=target, rawPredictionCol='prediction', metricName='areaUnderROC')
        metrics = evaluator.evaluate(dfpred)
        print(f'{params}, metrics: {metrics}')
        return -metrics

# COMMAND ----------

num_evals = 4
trials = Trials()
best_hyperparam = fmin(fn=train_models, space=search_space, algo=tpe.suggest, max_evals=num_evals, trials=trials, rstate=default_rng(random_seed))
print(best_hyperparam)

# COMMAND ----------

best_hyperparam

# COMMAND ----------

mlflow.autolog()
with mlflow.start_run(run_name='best-model-xgb') as run:
    va = VectorAssembler(inputCols=input_features, outputCol='features')
    estimator = XgboostClassifier(labelCol=target, random_state=random_seed, 
                                  learning_rate=best_hyperparam['learning_rate'], max_depth=int(best_hyperparam['max_depth']), n_estimators=int(best_hyperparam['n_estimators']))
    pl = Pipeline(stages=[va, estimator])
    plmodel = pl.fit(dftrain)
    # log model
    model_path = 'xgb-model'
    mlflow.spark.log_model(plmodel, model_path, input_example=dftrain.limit(5).toPandas())
    # prediction
    dfpred = plmodel.transform(dftest)
    # calculate metrics
    evaluator = BinaryClassificationEvaluator(labelCol=target, rawPredictionCol='prediction', metricName='areaUnderROC')
    metrics = evaluator.evaluate(dfpred)

# COMMAND ----------

run.info

# COMMAND ----------

# Register model
registered_model_uri = f'runs:/{run.info.run_id}/xgb-model'
registered_model_name = 'predict_wine_quality'
registered_model = mlflow.register_model(model_uri=registered_model_uri, name=registered_model_name)

# COMMAND ----------

registered_model

# COMMAND ----------

mlclient = mlflow.tracking.client.MlflowClient()

# COMMAND ----------

# Tracking registered model
model_details = mlclient.get_model_version(name=registered_model_name, version=2)
model_details

# COMMAND ----------

mlclient.update_registered_model(name=registered_model_name, description='XGB model to predict wine quality')

# COMMAND ----------

mlclient.transition_model_version_stage(name=model_details.name, version=2, stage='Production')

# COMMAND ----------

mlclient.get_model_version(name=model_details.name, version=model_details.version).current_stage

# COMMAND ----------


