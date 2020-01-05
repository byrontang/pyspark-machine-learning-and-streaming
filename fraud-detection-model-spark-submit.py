from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import col

from pyspark.ml.feature import (VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == "__main__":
    
    sc = SparkContext(appName="FraudDetection").getOrCreate()
    sess = SparkSession(sc)
    df = sess.read.csv("/user/edureka_854312/module_11_2/PS_20174392719_1491204439457_log.csv", 
                   inferSchema=True, 
                   header=True)
    df_model = df.select(['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                      'oldbalancedest', 'newbalanceDest', 'isFraud']).withColumn("isFraud", 
                                                                                 col("isFraud").cast("double"))
    
    type_indexer = StringIndexer(inputCol='type',outputCol='typeIndex')
    type_encoder = OneHotEncoder(inputCol='typeIndex',outputCol='typeVec')
    assembler = VectorAssembler(inputCols=['typeVec', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                           'oldbalancedest', 'newbalanceDest'],
                                outputCol='features')
    train_data, test_data = df_model.randomSplit([0.7,0.3])
    
    rf = RandomForestClassifier().setLabelCol('isFraud').setFeaturesCol('features').setNumTrees(10)
    pipeline = Pipeline(stages=[type_indexer, type_encoder, assembler, rf])
    model = pipeline.fit(train_data)
    results = model.transform(test_data)
    
    my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='isFraud')
    AUC = my_eval.evaluate(results)
    print("AUC:", AUC)

    model.save('/user/edureka_854312/module_11_2/model/')