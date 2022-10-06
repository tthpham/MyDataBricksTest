# Databricks notebook source
pip install scikit-surprise

# COMMAND ----------

from surprise import Reader, Dataset, SVD, NMF
from surprise.model_selection import cross_validate
# from pyspark.sql import sqlContext

# COMMAND ----------

random_seed = 27

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM tthpham2710.ratings

# COMMAND ----------

df_ratings = sqlContext.sql('select * from tthpham2710.ratings')

# COMMAND ----------

# df_ratings = _sqldf
dftrain, dftest = df_ratings.randomSplit([.8, .2], seed=random_seed)

# COMMAND ----------

display(df_ratings)

# COMMAND ----------

# Singular Value Decomposition
svd = SVD(random_state=random_seed)
# Non negative Matrix Factorization
nmf = NMF(random_state=random_seed)

# COMMAND ----------

reader = Reader(rating_scale=(0.5, 5.0))

# COMMAND ----------

# The columns must correspond to user id, item id and ratings
data_train = Dataset.load_from_df(dftrain.select(['userId', 'movieId', 'rating']).toPandas(), reader)
# data_test = Dataset.load_from_df(dftest.select(['userId', 'movieId', 'rating']).toPandas(), reader)

# COMMAND ----------

cross_validate(svd, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# COMMAND ----------

display(dftest)

# COMMAND ----------

predictions = svd.test(dftest.select(['userId', 'movieId', 'rating']).toPandas().values)

# COMMAND ----------

rating_pred = [p.est for p in predictions]

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as f
from pyspark.ml.linalg import DenseVector

# COMMAND ----------

DenseVector(rating_pred)DenseVector(rating_pred)

# COMMAND ----------


