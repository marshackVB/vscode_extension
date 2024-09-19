import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from pyspark.sql import SparkSession

from main.custom_funcs import get_spark_dataframe, transform_data, get_pipeline

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/marshall_carter")


spark = SparkSession.builder.getOrCreate()


raw_data = get_spark_dataframe()
features = transform_data(raw_data)

delta_location = 'main.default.mlc_titanic_features'
features.write.mode('overwrite').format('delta').saveAsTable(delta_location)

training_df = spark.table(delta_location).toPandas()
training_df.head()

with mlflow.start_run(run_name='xgboost') as run:

    run_id = run.info.run_id

    mlflow.autolog(log_input_examples=True,
                 log_model_signatures=True,
                 log_models=True,
                 silent=True)
    
    label = 'Survived'
    features = [col for col in training_df.columns if col not in [label, 'PassengerId']]

    X_train, X_val, y_train, y_val = train_test_split(training_df[features], 
                                                    training_df[label], 
                                                    test_size=0.25, 
                                                    random_state=123, 
                                                    shuffle=True)

    preprocessing_pipeline = get_pipeline()
    model = xgb.XGBClassifier(n_estimators = 25)

    classification_pipeline = Pipeline([("preprocess", preprocessing_pipeline), ("classifier", model)])
    classification_pipeline.fit(X_train, y_train)

    print(f"Training model with run id: {run_id}")
