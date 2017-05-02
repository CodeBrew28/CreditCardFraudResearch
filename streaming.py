import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors 
from pyspark.mllib.regression import LabeledPoint 
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
import pandas as pd
import csv
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel


def clean(x):
    if (x[29] != "Amount"):
        return x

def normalize(x):
    return LabeledPoint(float(x[30]), [float(x[0]), float(x[29])/ 25691.16])

conf = SparkConf().setAppName("LogisticRegression")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


# Load and parse the data
rdd = sc.textFile("file:///mnt/vdatanodea/datasets/creditcards/creditcard.csv")
data = rdd.mapPartitions(lambda x: csv.reader(x))
data = data.map( lambda x: clean(x) )
data = data.filter(lambda x: x != None)
normalizedData = data.map(normalize)

#split the training and test data
(trainingData, testData) = normalizedData.randomSplit([0.7, 0.3])
sample = sc.parallelize(trainingData.take(180000))
testsample = sc.parallelize(testData.take(78000))
lr = LogisticRegressionWithLBFGS.train(traitraningData)

conf = (SparkConf()
     .setMaster("local")
     .setAppName("My app")
     .set("spark.executor.memory", "1g"))

sc = SparkContext(conf = conf)

ssc = StreamingContext(sc, 1)
lines1 = ssc.textFileStream("file:///mnt/vdatanodea/datasets/creditcards/credit/b")
trainingData = lines1.map(lambda line: LabeledPoint( float(line.split(" ")[1]), [ (line.split(" ") [0]) ,  (line.split(" ") [2]) ])).cache()
trainingData.pprint()

lines2 = ssc.textFileStream("file:///mnt/vdatanodea/datasets/creditcards/credit/c")
testData = lines2.map(lambda line: LabeledPoint( float(line.split(" ")[1]), [ (line.split(" ") [0]) ,  (line.split(" ") [2]) ])).cache()
testData.pprint()

def handle_rdd(rdd):
    count = 0
    total = 0
    for r in rdd.collect():
        print( r.map(lambda p: (p.label, p.features, lr.predict(p.features))) )
        total = x.filter(lambda d: d[0] != d[1]).count() / float(testData.count())
        count += 1
    print(total/count)
labelsAndPreds = testData.transform(lambda rdd: handle_rdd)

labelsAndPreds.pprint()
ssc.start() 

# labelsAndPreds = testsample.map(lambda p: (p.label, p.features, lr.predict(p.features)))
# testErr = labelsAndPredictions.filter(lambda d: d[0] != d[1]).count() / float(testData.count())
# print("Training Error = " + str(trainErr))
# ssc.start() 







