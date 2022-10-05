# Databricks notebook source
# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06.csv"

# COMMAND ----------

raw_df = spark.read.csv(file_path, header="true", multiLine="true", inferSchema="true", escape='"')
display(raw_df)

# COMMAND ----------

columns_to_keep = [
    "host_is_superhost",
    "cancellation_policy",
    "instant_bookable",
    "host_total_listings_count",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "bed_type",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "price"
]

# COMMAND ----------

base_df = raw_df.select(columns_to_keep)

# COMMAND ----------

base_df.cache().count()

# COMMAND ----------

display(base_df)

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id, translate
fixed_price_df = base_df.withColumn("price", translate(col("price"), "$", "").cast("double"))
display(fixed_price_df)

# COMMAND ----------

fixed_price_df = fixed_price_df.withColumn("index", monotonically_increasing_id())

# COMMAND ----------

fixed_price_df.schema

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Store: Create a database for feature tables

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS my_features

# COMMAND ----------

# Create Feature Store client
from databricks.feature_store import FeatureStoreClient

fsclient = FeatureStoreClient()

# COMMAND ----------

fsclient.create_table(name="my_features.dataset_airbnb", primary_keys="index", schema=fixed_price_df.schema, description="Airbnb dataset", df=fixed_price_df)

# COMMAND ----------

fsclient.write_table(name="my_features.dataset_airbnb", df=fixed_price_df, mode="overwrite")

# COMMAND ----------

display(fsclient.read_table(name="my_features.dataset_airbnb"))

# COMMAND ----------

fsclient.get_table(name="my_features.dataset_airbnb").path_data_sources

# COMMAND ----------


