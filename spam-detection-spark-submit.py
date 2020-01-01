from __future__ import print_function

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession, SQLContext, Row
import datetime

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, CountVectorizerModel, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

if __name__ == "__main__":

    # Modify the model application to train the model and persist it
    sc = SparkContext(appName="SpamDetection")
    sess = SparkSession(sc)
    #spark = SparkSession.builder.appName("SpamDetection Notebook").getOrCreate()
        # Read training data
    raw = sess.read.option("delimiter","\t").csv("/user/edureka_854312/module_11_1/SMSSpamCollection").toDF("spam","message")
        # Build Pipeline
    tokenizer = Tokenizer().setInputCol("message").setOutputCol("words")
    stopwords = StopWordsRemover().getStopWords() + ["-"]
    remover = StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
    cvmodel = CountVectorizer().setInputCol("filtered").setOutputCol("features")
    indexer = StringIndexer().setInputCol("spam").setOutputCol("label")
    
    lr = LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    pipeline = Pipeline().setStages([tokenizer, remover, cvmodel, indexer, lr])
    model = pipeline.fit(raw)
    
    # Spark Streaming
    ssc = StreamingContext(sc, 60)
    
    now = datetime.datetime.now()
    filepath = "/user/edureka_854312/module_11_1/" + now.strftime("%Y-%m-%d/")
    print("filepath:", filepath)
    lines = ssc.textFileStream(filepath)

    def getSparkSessionInstance(sparkConf):
        if ("sparkSessionSingletonInstance" not in globals()):
            globals()["sparkSessionSingletonInstance"] = SparkSession \
                .builder \
                .config(conf=sparkConf) \
                .getOrCreate()
        return globals()["sparkSessionSingletonInstance"]

    def process(t, rdd):
        if rdd.isEmpty():
            print("filepath:", filepath)
            print("==== EMPTY ====")
            return

        print("=== RDD Found ===")
        rowRdd = rdd.map(lambda x: Row(message=x))
        spark = getSparkSessionInstance(rdd.context.getConf())
        df = spark.createDataFrame(rowRdd)
        print("DataFrame:")
        print(df.show())
        
        # Predict the SPAM messages and print the SPAM in the logs
        predictions = model.transform(df)
        print(predictions.show())
    
    lines.pprint()  
    lines.foreachRDD(process)
    
    ssc.start()
    ssc.awaitTermination()

    
    