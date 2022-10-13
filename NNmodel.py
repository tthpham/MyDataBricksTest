# Databricks notebook source
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# COMMAND ----------

random_seed = 27

# COMMAND ----------

# Load California Housing dataset from scikit-learn
cal_housing = fetch_california_housing()

# COMMAND ----------

cal_housing

# COMMAND ----------

# Split train and test data
Xtrain, Xtest, ytrain, ytest = train_test_split(cal_housing.data, cal_housing.target, test_size=0.2, random_state=random_seed)

# COMMAND ----------

nbrecords, nbfeatures = cal_housing.data.shape

# COMMAND ----------

# Create model and view TensorBoard in notebook
def create_model():
    model = Sequential()
    model.add(Dense(20, input_dim=nbfeatures, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

# COMMAND ----------

model = create_model()
model.compile(loss='mse', optimizer='Adam', metrics=['mse'])

# COMMAND ----------

experiment_log_dir = "/dbfs/tthpham2710/tb"
checkpoint_path = "/dbfs/tthpham2710/keras_checkpoint_weights.ckpt"

# COMMAND ----------

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)

# COMMAND ----------

history = model.fit(Xtrain, ytrain, validation_split=.2, epochs=35, callbacks=[tensorboard_callback, model_checkpoint, early_stopping])

# COMMAND ----------

model.evaluate(Xtest, ytest)

# COMMAND ----------


