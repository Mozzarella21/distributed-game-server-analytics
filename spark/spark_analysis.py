
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

spark = (
    SparkSession.builder
    .appName("GameServerSparkSQLAnalysis")
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

input_path = "hdfs:///game_logs/full_input/part-*.csv"
output_base = "hdfs:///game_logs/spark_results/sql"

df = (
    spark.read
    .schema(schema)
    .option("header", "false")
    .csv(input_path)
    .withColumn(
        "event_hour",
        F.hour(F.to_timestamp("timestamp"))
    )
)

total_records = df.count()
print(f"Total records: {total_records}")

# 1. Server activity analysis
server_load = (
    df.groupBy("server_id")
    .agg(F.count("*").alias("total_events"))
    .orderBy(F.desc("total_events"))
)

# 2. Regional latency analysis
regional_latency = (
    df.groupBy("server_id", "region")
    .agg(
        F.round(F.avg("latency_ms"), 2)
         .alias("avg_latency_ms"),
        F.count("*").alias("event_count")
    )
)

# 3. Disconnect analysis
disconnects = (
    df.groupBy("server_id")
    .agg(
        F.count("*").alias("total_events"),
        F.sum(
            F.when(
                F.upper(F.col("event_type")) == "DISCONNECT",
                1
            ).otherwise(0)
        ).alias("disconnects")
    )
    .withColumn(
        "disconnect_rate_pct",
        F.round(
            F.col("disconnects") /
            F.col("total_events") * 100,
            2
        )
    )
)

# 4. Peak gaming hours
peak_hours = (
    df.groupBy("event_hour")
    .agg(F.count("*").alias("total_events"))
    .orderBy("event_hour")
)

results = {
    "server_load": server_load,
    "regional_latency": regional_latency,
    "disconnects": disconnects,
    "peak_hours": peak_hours
}

for name, result in results.items():
    output_path = f"{output_base}/{name}"
    print(f"Writing {name} to {output_path}")

    (
        result.write
        .mode("errorifexists")
        .option("header", True)
        .csv(output_path)
    )

print("Spark SQL analysis completed successfully.")
spark.stop()
