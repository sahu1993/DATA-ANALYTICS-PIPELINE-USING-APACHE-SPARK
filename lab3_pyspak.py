from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import HashingTF, IDF, NGram
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from operator import add
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as f
import sys
import re

def tokens(context):
    file = context[0]
    words = re.sub('[^a-z0-9]+',' ',context[1].lower()).split()
    file = file.split("/")[-1]
    #Class Label 0 - Sports, 1 - Politics, 2- Business, 3- Education
    if(re.match( r'Sportsfile.*', file)):
        return (0.0, file ,words)
    elif(re.match( r'Politicsfile.*', file)):
        return (1.0, file, words)
    elif(re.match( r'Businessfile.*', file)):
        return (2.0, file, words)
    else:
        return (3.0, file, words)

if __name__ == "__main__":
    conf = SparkConf()
    conf.setAppName( "part1" )
    conf.set("spark.executor.memory", "2g")
    sc = SparkContext.getOrCreate(conf = conf)
    #reading input
    lines =sc.wholeTextFiles("data/Sports")
    #configuring SparkSession
    spark=SparkSession(sc)

    hasattr(lines, "toDF")
    #tokeinizing the words and converting into dataframes
    tokenizeDf0 = lines.map(tokens).toDF(["label", "fileName", "words"])

    lines =sc.wholeTextFiles("data/Politics")
    hasattr(lines, "toDF")
    tokenizeDf1 = lines.map(tokens).toDF(["label", "fileName", "words"])

    lines =sc.wholeTextFiles("data/Business")
    hasattr(lines, "toDF")
    tokenizeDf2 = lines.map(tokens).toDF(["label", "fileName", "words"])

    lines =sc.wholeTextFiles("data/Education")
    hasattr(lines, "toDF")
    tokenizeDf3 = lines.map(tokens).toDF(["label", "fileName", "words"])

    result_tokenize_data_1 = tokenizeDf0.union(tokenizeDf1)
    result_tokenize_data_2 = tokenizeDf2.union(tokenizeDf3)
    tokenizeDf = result_tokenize_data_1.union(result_tokenize_data_2)

    #removing the Stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    filteredWordsDf = remover.transform(tokenizeDf)

    #finding the tf value
    hashingTF = HashingTF(inputCol = "filtered", outputCol = "rawFeatures")
    tf = hashingTF.transform(filteredWordsDf)

    #finding the idf value
    idf = IDF(inputCol = "rawFeatures", outputCol = "features", )
    idfModel = idf.fit(tf)
    rescaledData = idfModel.transform(tf)

    (trainingDataLR, testDataLR) = rescaledData.randomSplit([0.8, 0.2], seed = 100)
    (trainingDataNB, testDataNB) = rescaledData.randomSplit([0.8, 0.2], seed = 100)

    lines =sc.wholeTextFiles("data/UnknownSetData")
    hasattr(lines, "toDF")
    tokenizeDf4 = lines.map(tokens).toDF(["label", "fileName", "words"])

    #removing the Stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    filteredWordsDf = remover.transform(tokenizeDf4)

    #finding the tf value
    hashingTF = HashingTF(inputCol = "filtered", outputCol = "rawFeatures")
    tf = hashingTF.transform(filteredWordsDf)

    #finding the idf value
    idf = IDF(inputCol = "rawFeatures", outputCol = "features", )
    idfModel = idf.fit(tf)

    testDataUnknownSetLR = idfModel.transform(tf)
    testDataUnknownSetRF = idfModel.transform(tf)
    testDataUnknownSetNB = idfModel.transform(tf)


    #### Machine Learning####
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    logisticRegressionModel = lr.fit(trainingDataLR)

    predictions = logisticRegressionModel.transform(testDataLR)
    #predictions.select("fileName","probability","label","prediction").show()
    #predictions.select("fileName","probability","label","prediction").rdd.saveAsTextFile("data/output")

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print("AccuracyLRTest  = %g" % (accuracy))

    predictionsLabels = predictions.select("prediction","label").rdd
    metrics = MulticlassMetrics(predictionsLabels)
    confusionMatrix = metrics.confusionMatrix().toArray()
    print(confusionMatrix)

    predictions = logisticRegressionModel.transform(testDataUnknownSetLR)
    #predictions.select("fileName","probability","label","prediction").show()
    #predictions.select("fileName","probability","label","prediction").rdd.saveAsTextFile("data/output")

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print("AccuracyLRTestUnknow  = %g" % (accuracy))

    predictionsLabels = predictions.select("prediction","label").rdd
    metrics = MulticlassMetrics(predictionsLabels)
    confusionMatrix = metrics.confusionMatrix().toArray()
    print(confusionMatrix)

    nb = NaiveBayes(smoothing=1)
    naiveBayesModel = nb.fit(trainingDataNB)

    predictions = naiveBayesModel.transform(testDataNB)
    #predictions.select("fileName","probability","label","prediction").show()
    #predictions.select("fileName","probability","label","prediction").rdd.saveAsTextFile("data/outputNB1")

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print("AccuracyNBTest  = %g" % (accuracy))

    predictionsLabels = predictions.select("prediction","label").rdd
    metrics = MulticlassMetrics(predictionsLabels)
    confusionMatrix = metrics.confusionMatrix().toArray()
    print(confusionMatrix)

    predictions = naiveBayesModel.transform(testDataUnknownSetNB)
    #predictions.select("fileName","probability","label","prediction").show()
    #predictions.select("fileName","probability","label","prediction").rdd.saveAsTextFile("data/outputNB1")

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print("AccuracyNBTestUnknown  = %g" % (accuracy))

    predictionsLabels = predictions.select("prediction","label").rdd
    metrics = MulticlassMetrics(predictionsLabels)
    confusionMatrix = metrics.confusionMatrix().toArray()
    print(confusionMatrix)

    sc.stop()
