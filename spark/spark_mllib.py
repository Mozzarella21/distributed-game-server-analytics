
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml import Pipeline

spark = (
    SparkSession.builder
    .appName("GameServerDisconnectPrediction")
    .getOrCreate()
)

schema = StructType([
    StructField("timestamp", StringType()),
    StructField("server_id", StringType()),
    StructField("region", StringType()),
    StructField("player_id", StringType()),
    StructField("session_id", StringType()),
    StructField("event_type", StringType()),
    StructField("latency_ms", DoubleType()),
    StructField("packet_loss_pct", DoubleType()),
    StructField("jitter_ms", DoubleType()),
    StructField("game_mode", StringType()),
    StructField("server_load_pct", DoubleType())
])

df = (
    spark.read
    .schema(schema)
    .option("header", "false")
    .csv("hdfs:///game_logs/full_input/part-*.csv")
)

features = [
    "latency_ms",
    "packet_loss_pct",
    "jitter_ms",
    "server_load_pct"
]

df = (
    df.withColumn(
        "label",
        F.when(
            F.upper(F.col("event_type")) == "DISCONNECT", 1.0
        ).otherwise(0.0)
    )
    .dropna(subset=features + ["label", "session_id"])
)

# Split by session to reduce leakage between train and test.
df = df.withColumn(
    "split_bucket",
    F.pmod(F.xxhash64("session_id"), F.lit(10))
)

train = df.filter(F.col("split_bucket") < 8)
test = df.filter(F.col("split_bucket") >= 8)

assembler = VectorAssembler(
    inputCols=features,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=False
)

classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=30,
    regParam=0.05,
    threshold=0.01
)

pipeline = Pipeline(stages=[assembler, scaler, classifier])

print("Training disconnect prediction model...")
model = pipeline.fit(train)

print("Evaluating model...")
predictions = model.transform(test).cache()

binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = binary_evaluator.evaluate(predictions)

metrics = {}
for metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric
    )
    metrics[metric] = evaluator.evaluate(predictions)

print("ROC-AUC:", round(auc, 4))
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Save metrics and confusion matrix.
output_base = "hdfs:///game_logs/spark_results/ml_threshold_001"

metric_rows = [("roc_auc", float(auc))]
metric_rows += [(k, float(v)) for k, v in metrics.items()]

spark.createDataFrame(
    metric_rows, ["metric", "value"]
).coalesce(1).write.mode("errorifexists").option(
    "header", True
).csv(output_base + "/metrics")

confusion = (
    predictions.groupBy("label", "prediction")
    .count()
)

confusion.write.mode("errorifexists").option(
    "header", True
).csv(output_base + "/confusion_matrix")

predictions.unpersist()
spark.stop()

print("Spark MLlib analysis completed successfully.")
