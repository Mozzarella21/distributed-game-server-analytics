
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Game Server Analytics",
    page_icon="🎮",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "final_results"


@st.cache_data
def load_csv(filename):
    return pd.read_csv(DATA / filename,
        keep_default_na=False)


server = load_csv("server_load.csv")
latency = load_csv("regional_latency.csv")
disconnects = load_csv("disconnects.csv")
peak = load_csv("peak_hours.csv")

baseline = load_csv("ml_metrics_baseline.csv")
updated = load_csv("ml_metrics_updated.csv")

baseline_cm = load_csv("confusion_matrix_baseline.csv")
updated_cm = load_csv("confusion_matrix_updated.csv")


def metric_value(df, name):
    return float(
        df.loc[df["metric"] == name, "value"].iloc[0]
    )


def confusion_stats(df):
    counts = {
        (int(row.label), int(row.prediction)): int(row["count"])
        for _, row in df.iterrows()
    }

    tn = counts.get((0, 0), 0)
    fp = counts.get((0, 1), 0)
    fn = counts.get((1, 0), 0)
    tp = counts.get((1, 1), 0)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return tn, fp, fn, tp, precision, recall, f1


st.title("Online Game Server Log Analytics")
st.caption(
    "Distributed Hadoop HDFS + Apache Spark SQL + Spark MLlib"
)

st.info(
    "Dataset: 10 million synthetic game-server events. "
    "ML predictions represent simulated disconnect events."
)

total_events = int(server["total_events"].sum())
total_servers = server["server_id"].nunique()
total_disconnects = int(disconnects["disconnects"].sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Events", f"{total_events:,}")
c2.metric("Game Servers", f"{total_servers:,}")
c3.metric("Disconnect Events", f"{total_disconnects:,}")
c4.metric(
    "Overall Disconnect Rate",
    f"{100 * total_disconnects / total_events:.3f}%"
)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Server Activity",
    "Regional Latency",
    "Disconnect Analysis",
    "Peak Gaming Hours",
    "MLlib Prediction"
])

with tab1:
    st.subheader("Server Activity Analysis")

    top = server.nlargest(20, "total_events")

    fig = px.bar(
        top,
        x="server_id",
        y="total_events",
        title="Top 20 Servers by Event Count",
        labels={
            "server_id": "Server",
            "total_events": "Events"
        }
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(server, use_container_width=True)

with tab2:
    st.subheader("Regional Latency Analysis")

    regions = sorted(
    latency["region"].dropna().astype(str).unique()
)

    selected_region = st.selectbox(
        "Select Region",
        regions
    )

    filtered = latency[
    latency["region"].astype(str) == selected_region
].sort_values("avg_latency_ms", ascending=False)

    fig = px.bar(
        filtered.head(20),
        x="server_id",
        y="avg_latency_ms",
        title=f"Highest Average Latency: {selected_region}",
        labels={
            "server_id": "Server",
            "avg_latency_ms": "Average Latency (ms)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(filtered, use_container_width=True)

with tab3:
    st.subheader("Disconnect Analysis")

    top_disconnects = disconnects.nlargest(
        20, "disconnect_rate_pct"
    )

    fig = px.bar(
        top_disconnects,
        x="server_id",
        y="disconnect_rate_pct",
        title="Top 20 Servers by Disconnect Event Rate",
        labels={
            "server_id": "Server",
            "disconnect_rate_pct": "Disconnect Rate (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(disconnects, use_container_width=True)

with tab4:
    st.subheader("Peak Gaming Hours")

    peak = peak.sort_values("event_hour")

    fig = px.line(
        peak,
        x="event_hour",
        y="total_events",
        markers=True,
        title="Game Events by Hour of Day",
        labels={
            "event_hour": "Hour",
            "total_events": "Events"
        }
    )

    fig.update_xaxes(dtick=1)

    st.plotly_chart(fig, use_container_width=True)

    busiest = peak.loc[peak["total_events"].idxmax()]

    st.metric(
        "Busiest Hour",
        f"{int(busiest['event_hour']):02d}:00",
        f"{int(busiest['total_events']):,} events",
        delta_color="off"
    )

with tab5:
    st.subheader("Spark MLlib Disconnect Prediction")

    models = {
        "Baseline (threshold 0.5)": (
            baseline, baseline_cm
        ),
        "Updated (threshold 0.01)": (
            updated, updated_cm
        )
    }

    selected = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    metrics, cm = models[selected]

    tn, fp, fn, tp, precision, recall, f1 = (
        confusion_stats(cm)
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Accuracy",
        f"{metric_value(metrics, 'accuracy'):.2%}"
    )
    b.metric(
        "ROC-AUC",
        f"{metric_value(metrics, 'roc_auc'):.4f}"
    )
    c.metric(
        "Disconnect Precision",
        f"{precision:.2%}"
    )
    d.metric(
        "Disconnect Recall",
        f"{recall:.2%}"
    )

    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Normal", "Actual Disconnect"],
        columns=["Predicted Normal", "Predicted Disconnect"]
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric("Disconnect-Class F1", f"{f1:.4f}")

    st.write(
        "The baseline model predicted no disconnects. "
        "The lower-threshold model detects some disconnects "
        "but produces more false positives."
    )

    comparison = []
    for name, (m, c) in models.items():
        stats = confusion_stats(c)
        comparison.append({
            "Model": name,
            "Accuracy": metric_value(m, "accuracy"),
            "ROC-AUC": metric_value(m, "roc_auc"),
            "Disconnect Precision": stats[4],
            "Disconnect Recall": stats[5],
            "Disconnect F1": stats[6],
            "False Positives": stats[1],
            "False Negatives": stats[2]
        })

    st.dataframe(
        pd.DataFrame(comparison),
        use_container_width=True
    )

st.divider()

st.caption(
    "Project: Distributed Online Game Server Log Analysis "
    "using Hadoop HDFS, YARN, Spark SQL and Spark MLlib."
)
