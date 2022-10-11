# Databricks notebook source
pip install scikit-surprise

# COMMAND ----------

from surprise import Reader, Dataset, SVD, NMF
from surprise.model_selection import cross_validate
import mlflow
from pandas import DataFrame as pdDataFrame
from pyspark.sql import SparkSession
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from numpy import mean as npmean

# COMMAND ----------

random_seed = 27

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM tthpham2710.ratings

# COMMAND ----------

df_ratings = sqlContext.sql('select * from tthpham2710.ratings')
dftrain, dftest = df_ratings.randomSplit([.8, .2], seed=random_seed)

# COMMAND ----------

display(df_ratings)

# COMMAND ----------

# Singular Value Decomposition
svd = SVD(random_state=random_seed)
# Non negative Matrix Factorization
nmf = NMF(random_state=random_seed)

# COMMAND ----------

# Prepare data for train
reader = Reader(rating_scale=(0.5, 5.0))
data_train = Dataset.load_from_df(dftrain.select(['userId', 'movieId', 'rating']).toPandas(), reader)

# COMMAND ----------

algo = svd  # define algorithm to use
nb_folds= 5
# Create run
with mlflow.start_run(run_name='Create-movie-recommandation-model') as my_run:
    # log parameters
    mlflow.log_params({
        'random seed': random_seed,
        'algorithm': algo,
        'nb folds': nb_folds
    })
    metrics = cross_validate(algo, data_train, measures=['RMSE', 'MAE'], cv=nb_folds, verbose=True)
    # log metrics train
    mlflow.log_metrics({
        'train_kfolds_rmse': npmean(metrics['test_rmse']),
        'train_kfolds_mae': npmean(metrics['test_mae'])
    })
    # make prediction on test dataset
    predictions = algo.test(dftest.select(['userId', 'movieId', 'rating']).toPandas().values)
    dfpred = dftest.toPandas()    
    dfpred['pred'] = [p.est for p in predictions]
    r2, mape, mae = r2_score(dfpred.rating, dfpred.pred), mean_absolute_percentage_error(dfpred.rating, dfpred.pred), mean_absolute_error(dfpred.rating, dfpred.pred)
    # log metrics test
    mlflow.log_metrics({
        'test_mae': mae,
        'test_r2': r2,
        'test_mape': mape
    })

# COMMAND ----------

rs = cross_validate(svd, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# COMMAND ----------

rs['test_rmse']

# COMMAND ----------

predictions = algo.test(dftest.select(['userId', 'movieId', 'rating']).toPandas().values)

# COMMAND ----------

rating_pred = [p.est for p in predictions]

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as f
from pyspark.ml.linalg import DenseVector
from pandas import DataFrame as pdDataFrame
from pyspark.sql import SparkSession
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# COMMAND ----------

dfpred = dftest.toPandas()

# COMMAND ----------

dfpred['pred'] = rating_pred

# COMMAND ----------

display(dfpred)

# COMMAND ----------

r2_score(dfpred.rating, dfpred.pred)

# COMMAND ----------

mean_absolute_percentage_error(dfpred.rating, dfpred.pred)

# COMMAND ----------

mean_absolute_error(dfpred.rating, dfpred.pred)

# COMMAND ----------


