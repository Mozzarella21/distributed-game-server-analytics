# Distributed Multiplayer Game Server Log Analytics and Disconnect Prediction

A distributed big-data project that processes 10 million synthetic multiplayer game-server events using Apache Hadoop, MapReduce, Apache Spark and Spark MLlib. The project analyzes server activity, regional latency, disconnections and peak gaming hours, and evaluates Logistic Regression models for disconnect classification.

## 1. Project overview

Multiplayer gaming platforms generate large volumes of event logs containing network measurements, player activity and server conditions. Analyzing these records can help identify traffic patterns, network performance differences and unusual rates of disconnection.

This project implements a three-node Hadoop cluster connected through Tailscale. HDFS stores the generated dataset, Hadoop MapReduce provides batch-processing implementations, Spark SQL performs distributed aggregations, and Spark MLlib trains a supervised disconnect classifier.

A Streamlit dashboard presents the aggregated results and model evaluation metrics.

## 2. Technology stack

| Component | Technology |
|---|---|
| Operating environment | Ubuntu 24.04 under Windows Subsystem for Linux 2 |
| Distributed storage | Apache Hadoop HDFS 3.4.2 |
| Resource management | Hadoop YARN |
| Distributed analytics | Apache Spark 3.5.x |
| Machine learning | Spark MLlib Logistic Regression |
| Programming language | Python 3.12 |
| Visualization | Streamlit, Pandas and Plotly |
| Cluster networking | Tailscale |
| Version control | Git and GitHub |

## 3. Distributed cluster architecture

The project uses three laptops connected through a private Tailscale network.

| Node | Role | Services |
|---|---|---|
| Shivom's laptop | Hadoop master | NameNode, ResourceManager, DataNode, NodeManager |
| Sabad's laptop | Worker 1 and Spark submission client | DataNode, NodeManager, Spark driver |
| Sannidhya's laptop | Worker 2 | DataNode, NodeManager |

The cluster uses HDFS replication factor 2 for the primary dataset.

Hadoop YARN schedules distributed jobs, while Spark executors process data on available worker nodes.

The implementation was tested using three live DataNodes and three registered YARN NodeManagers.

## 4. Dataset

The dataset was generated using `data/generate_logs.py`.

**Dataset characteristics**

- Total records: 10,000,000
- Data format: CSV
- Generated files: 32
- Time range: 90 simulated days beginning January 1, 2026
- Server count: 100
- Geographic regions: ASIA, EUROPE, NA, SA and OCEANIA
- HDFS input path: `/game_logs/full_input`
- Default random seed: `20261105`

Each record contains the following 11 fields:

| Field | Description |
|---|---|
| timestamp | Event timestamp |
| server_id | Game-server identifier |
| region | Geographic server region |
| player_id | Player identifier |
| session_id | Session identifier |
| event_type | Login, match, activity, logout or disconnect event |
| latency_ms | Simulated network latency in milliseconds |
| packet_loss_pct | Simulated packet-loss percentage |
| jitter_ms | Simulated network jitter in milliseconds |
| game_mode | Multiplayer game mode |
| server_load_pct | Simulated server load percentage |

The generator uses weighted gaming hours, region-dependent latency distributions, peak-hour server loads and simulated degraded network conditions.

Disconnect probability depends on simulated packet loss, session network penalties and server load, with a random component.

The dataset is synthetic. Findings from these experiments should not be interpreted as measurements from a production gaming platform.

## 5. Hadoop MapReduce analysis

Four MapReduce implementations are included in the `mapreduce/` directory.

### Server load analysis

Counts events for each server to compare recorded activity across the 100 servers.

### Regional latency analysis

Processes latency measurements to summarize network conditions across servers and regions.

### Disconnect analysis

Processes disconnect events to support server-level disconnection analysis.

### Peak gaming hours

Aggregates event activity by hour to identify periods with the highest number of recorded events.

Each implementation contains a Python mapper and reducer intended for Hadoop Streaming.

## 6. Spark SQL and DataFrame analysis

The Spark SQL implementation is available at:

`spark/spark_analysis.py`

It processes the complete 10-million-record dataset stored in HDFS.

The completed analysis produces four result datasets:

| Output | Description |
|---|---|
| server_load.csv | Event counts by server |
| regional_latency.csv | Average latency by server and region |
| disconnects.csv | Server-level disconnect statistics |
| peak_hours.csv | Event counts by hour |

The outputs contain 100, 100, 100 and 24 rows respectively.

Verification of the server-load output produced an aggregate total of exactly **10,000,000 events**.

The results are stored in `results/spark_sql/`.

## 7. Spark MLlib disconnect prediction

The project uses Spark MLlib Logistic Regression to classify whether individual log events are marked `DISCONNECT`.

The classifier uses four numerical features:

- Latency
- Packet loss
- Jitter
- Server load percentage

The model was evaluated using a deterministic session-based split to separate training and test sessions.

Two classification thresholds were tested: the default threshold of 0.50 and an experimental threshold of 0.01.

### Evaluation results

| Metric | Threshold 0.50 | Threshold 0.01 |
|---|---:|---:|
| Accuracy | 99.56% | 97.84% |
| ROC-AUC | 0.7199 | 0.7199 |
| Disconnect recall | 0% | 12.60% |
| Disconnect precision | 0%* | 3.05% |
| True positives | 0 | 1,110 |
| False positives | 0 | 35,301 |
| False negatives | 8,807 | 7,697 |
| True negatives | 1,986,123 | 1,950,822 |

*The baseline model made no positive predictions, so its positive-class precision is mathematically undefined and is displayed as zero by convention.*

### Interpretation

The baseline classifier achieved 99.56% accuracy but failed to identify any disconnect events.

This occurred because disconnects represented approximately 0.44% of the test records. Predicting every record as a normal event therefore produced high overall accuracy.

Lowering the classification threshold to 0.01 increased disconnect recall to 12.60%, but produced 35,301 false positives.

The ROC-AUC remained approximately 0.72 across both experiments, indicating that changing the threshold affected the final classifications rather than the underlying ranking performance.

These results show that overall accuracy alone is insufficient for evaluating rare-event detection.

The current classifier is an experimental baseline, not a production-ready disconnect prediction system.

## 8. K-Means experiment and implementation limitations

The original project synopsis proposed unsupervised anomaly detection using Spark MLlib K-Means.

A K-Means pipeline was developed to aggregate server conditions into time windows and identify unusual observations using cluster-centroid distances.

However, the distributed K-Means experiment did not produce validated final results.

The initial attempt experienced executor heartbeat timeouts and prolonged HDFS write acknowledgments. Later attempts encountered slow Spark library transfers and YARN ApplicationMaster registration timeouts.

The final attempted application failed after two ApplicationMaster timeouts.

The project therefore uses the completed Logistic Regression experiments as its final Spark MLlib analysis.

K-Means is documented as attempted work, not a successful anomaly-detection result.

## 9. Streamlit dashboard

The dashboard is implemented in `dashboard/app.py`.

It provides five analytical views:

1. Server activity
2. Regional latency
3. Disconnect analysis
4. Peak gaming hours
5. Logistic Regression model evaluation

The dashboard reads aggregated CSV results rather than processing the complete 10-million-record dataset at runtime.

### Run the dashboard

Install the required dependencies in a Python virtual environment:

```bash
python3 -m venv dashboard-env
source dashboard-env/bin/activate
pip install -r requirements.txt
```

Start Streamlit from the repository root:

```bash
streamlit run dashboard/app.py
```

Open the local address displayed by Streamlit, normally `http://localhost:8501`.

## 10. Reproducing the dataset

Generate a smaller sample:

```bash
python3 data/generate_logs.py \
  --records 100000 \
  --parts 4 \
  --output sample_logs
```

Generate the full dataset:

```bash
python3 data/generate_logs.py \
  --records 10000000 \
  --parts 32 \
  --seed 20261105 \
  --output game_logs
```

The generator writes records without CSV headers by default, matching the original Hadoop input format.

### Upload to HDFS

With a running Hadoop cluster:

```bash
hdfs dfs -mkdir -p /game_logs/full_input
hdfs dfs -put game_logs/part-*.csv /game_logs/full_input/
```

Verify:

```bash
hdfs dfs -ls /game_logs/full_input
hdfs fsck /game_logs/full_input
```

## 11. Running the Spark analyses

With a configured Spark-on-YARN cluster, submit the SQL analysis:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  spark/spark_analysis.py
```

Run the Logistic Regression experiment:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  spark/spark_mllib.py
```

Distributed execution requires the Hadoop and YARN configuration, Python dependencies and network connectivity to be correctly configured on the participating nodes.

The exact driver networking settings may differ between environments.

## 12. Project limitations

The implementation has several limitations.

The dataset is synthetic, and its statistical relationships are determined by the generator rather than observed production traffic.

The Logistic Regression model has low positive-class recall and precision. Its results should not be used to claim reliable real-world disconnect prediction.

The implemented dataset also differs from the original synopsis. It does not contain separate server CPU usage, server memory usage, concurrent-player counts or disconnect-reason fields.

Hourly event counts measure recorded activity, not concurrent players. The server-load percentage is a simulated combined load measure rather than separate CPU and memory measurements.

The K-Means anomaly-detection component was attempted but not successfully completed.

Performance comparisons should use measured execution times from equivalent workloads; no unverified benchmark values are included.

## 13. Repository contents

```text
data/                 Synthetic dataset generator
mapreduce/            Four Hadoop Streaming analyses
spark/                Spark SQL and Logistic Regression
dashboard/            Streamlit dashboard
results/spark_sql/    Aggregated SQL outputs
results/logistic_regression/  Model metrics and confusion matrices
screenshots/          Execution and dashboard evidence
report/               Final academic documentation
```

## 14. Conclusion

The project successfully processed 10 million synthetic multiplayer game-server records using Hadoop HDFS, YARN and Spark SQL.

Four analytical outputs were generated, and two Logistic Regression experiments were evaluated using Spark MLlib.

The experiments demonstrated that a classifier can achieve high overall accuracy while failing to identify rare disconnect events. Lowering the classification threshold improved recall but substantially increased false positives.

The project also documented practical limitations of a geographically distributed Hadoop cluster operating over Tailscale, including network-dependent HDFS performance and YARN application startup failures.
