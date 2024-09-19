from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import pyspark.sql.functions as func
from pyspark.sql.functions import col
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

spark = SparkSession.builder.getOrCreate()

def get_spark_dataframe(dbfs_csv_folder):
  """Load sample .csv file from Repo into a Spark DataFrame"""

  columns_and_types = [('PassengerId', StringType()),
                       ('Survived', IntegerType()), 
                       ('Pclass', StringType()),
                       ('Name', StringType()), 
                       ('Sex', StringType()), 
                       ('Age', DoubleType()), 
                       ('SibSp', StringType()),
                       ('Parch', StringType()), 
                       ('Ticket', StringType()), 
                       ('Fare', DoubleType()),
                       ('Cabin', StringType()),
                       ('Embarked', StringType())]

  spark_schema = StructType()
  for col_name, spark_type in columns_and_types:
      spark_schema.add(StructField(col_name, spark_type, True))

  df_spark = spark.read.format("csv").option("header", "true").schema(spark_schema).load(dbfs_csv_folder)

  return df_spark


def transform_data(raw_data):
    """Transform the raw data into features"""

    transformed_data = (raw_data.withColumn('NamePrefix', func.regexp_extract(col('Name'), '([A-Za-z]+)\.', 1))
                                .withColumn('NameSecondary_extract', func.regexp_extract(col('Name'), '\(([A-Za-z ]+)\)', 1))
                                .withColumn('TicketChars_extract', func.regexp_extract(col('Ticket'), '([A-Za-z]+)', 1))
                                .withColumn("CabinChar", func.split(col("Cabin"), '')[0])
                                .withColumn("CabinMulti_extract", func.size(func.split(col("Cabin"), ' ')))
                                .withColumn("FareRounded", func.round(col("Fare"), 0))
                        
                                .selectExpr("PassengerId",
                                            "Sex",
                                            "case when Age = 'NaN' then NULL else Age end as Age",
                                            "SibSp",
                                            "NamePrefix",
                                            "FareRounded",
                                            "CabinChar",
                                            "Embarked",
                                            "Parch",
                                            "Pclass",
                                            "case when length(NameSecondary_extract) > 0 then NameSecondary_extract else NULL end as NameSecondary",
                                            "case when length(TicketChars_extract) > 0 then upper(TicketChars_extract) else NULL end as TicketChars",
                                            "case when CabinMulti_extract < 0 then '0' else cast(CabinMulti_extract as string) end as CabinMulti",
                                            "Survived")
                    
                            .selectExpr("*",
                                        "case when NameSecondary is not NULL then '1' else '0' end as NameMultiple",
                                        "case when TicketChars is NULL then '1' else '0' end as MissingTicketChars")
                    
                            .drop("NameSecondary", "TicketChars"))
  
    return transformed_data


def get_pipeline():
    """
    Return a scikit-learn ColumnTranformer that performs feature pre-processing
    """

    categorical_vars = ['NamePrefix', 'Sex', 'CabinChar', 'CabinMulti', 'Embarked', 'Parch', 'Pclass', 'SibSp']
    numeric_vars = ['Age', 'FareRounded']
    binary_vars = ['NameMultiple', 'MissingTicketChars']

    # Create the a pre-processing and modleing pipeline
    binary_transform = make_pipeline(SimpleImputer(strategy = 'constant', fill_value = 'missing'))

    numeric_transform = make_pipeline(SimpleImputer(strategy = 'most_frequent'))

    categorical_transform = make_pipeline(SimpleImputer(missing_values = None, strategy = 'constant', fill_value = 'missing'), 
                                        OneHotEncoder(handle_unknown="ignore"))

    transformer = ColumnTransformer([('categorial_vars', categorical_transform, categorical_vars),
                                    ('numeric_vars', numeric_transform, numeric_vars),
                                    ('binary_vars', binary_transform, binary_vars)],
                                    remainder = 'drop')

    return transformer