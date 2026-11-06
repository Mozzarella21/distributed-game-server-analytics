
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Game Server Analytics",
    page_icon="🎮",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
SQL_DATA = ROOT / "results" / "spark_sql"
ML_DATA = ROOT / "results" / "logistic_regression"

REGION_NAMES = {
    "ASIA": "Asia",
    "EUROPE": "Europe",
    "NA": "North America",
    "SA": "South America",
    "OCEANIA": "Oceania"
}

@st.cache_data
def load_csv(filename):
    sql_files = {
        "server_load.csv",
        "regional_latency.csv",
        "disconnects.csv",
        "peak_hours.csv"
    }
    folder = SQL_DATA if filename in sql_files else ML_DATA
    return pd.read_csv(
        folder / filename,
        keep_default_na=False
    )

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
        (int(row["label"]), int(row["prediction"])):
        int(row["count"])
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


st.title("Multiplayer Game Server Analytics")
st.caption(
    "Hadoop HDFS | Spark SQL | Spark MLlib | "
    "Three-node distributed cluster"
)

st.info(
    "Analysis of 10 million synthetic multiplayer game-server "
    "events. All charts use saved distributed-processing results."
)

total_events = int(server["total_events"].sum())
total_servers = server["server_id"].nunique()
total_disconnects = int(disconnects["disconnects"].sum())
overall_rate = 100 * total_disconnects / total_events

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Events", f"{total_events:,}")
c2.metric("Game Servers", total_servers)
c3.metric("Disconnect Events", f"{total_disconnects:,}")
c4.metric("Disconnect Rate", f"{overall_rate:.3f}%")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Server Activity",
    "Regional Latency",
    "Disconnect Analysis",
    "Peak Gaming Hours",
    "MLlib Prediction"
])

# -------------------------------------------------
# TAB 1: SERVER ACTIVITY
# -------------------------------------------------

with tab1:
    st.subheader("Server Activity Analysis")

    top = server.nlargest(
        15, "total_events"
    ).sort_values("total_events")

    fig = px.bar(
        top,
        x="total_events",
        y="server_id",
        orientation="h",
        text="total_events",
        title="15 Most Active Game Servers",
        labels={
            "total_events": "Recorded Events",
            "server_id": "Server"
        },
        color="total_events",
        color_continuous_scale="Blues"
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )
    fig.update_layout(
        height=550,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_range=[
            0, top["total_events"].max() * 1.20
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    # Derive each server's region from its corresponding
    # server_id / region mapping in the latency output.
    mapping = latency[[
        "server_id", "region"
    ]].drop_duplicates("server_id")

    regional = (
        server.merge(mapping, on="server_id", how="left")
        .groupby("region", as_index=False)["total_events"]
        .sum()
    )

    regional["Region"] = regional["region"].map(
        REGION_NAMES
    ).fillna(regional["region"])

    fig = px.bar(
        regional.sort_values("total_events"),
        x="Region",
        y="total_events",
        color="Region",
        text="total_events",
        title="Total Event Activity by Region",
        labels={
            "total_events": "Events"
        }
    )
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        server.sort_values(
            "total_events", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

# -------------------------------------------------
# TAB 2: REGIONAL LATENCY
# -------------------------------------------------

with tab2:
    st.subheader("Regional Latency Analysis")

    # Weighted mean: use event_count so each record
    # contributes equally to the regional average.
    regional_latency = (
        latency.assign(
            weighted_latency=(
                latency["avg_latency_ms"]
                * latency["event_count"]
            )
        )
        .groupby("region", as_index=False)
        .agg(
            weighted_latency=("weighted_latency", "sum"),
            events=("event_count", "sum")
        )
    )

    regional_latency["avg_latency_ms"] = (
        regional_latency["weighted_latency"]
        / regional_latency["events"]
    )

    regional_latency["Region"] = (
        regional_latency["region"]
        .map(REGION_NAMES)
        .fillna(regional_latency["region"])
    )

    regional_latency = regional_latency.sort_values(
        "avg_latency_ms"
    )

    fig = px.bar(
        regional_latency,
        x="Region",
        y="avg_latency_ms",
        color="Region",
        text="avg_latency_ms",
        title="Average Network Latency by Region",
        labels={
            "avg_latency_ms": "Average Latency (ms)"
        }
    )
    fig.update_traces(
        texttemplate="%{text:.1f} ms",
        textposition="outside"
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    regions = sorted(latency["region"].unique())

    selected_region = st.selectbox(
        "Inspect servers in a region",
        regions,
        format_func=lambda r: REGION_NAMES.get(r, r)
    )

    filtered = latency[
        latency["region"] == selected_region
    ].sort_values("avg_latency_ms")

    fig = px.bar(
        filtered,
        x="server_id",
        y="avg_latency_ms",
        color="avg_latency_ms",
        color_continuous_scale="Oranges",
        title=(
            f"Server Latency: "
            f"{REGION_NAMES.get(selected_region, selected_region)}"
        ),
        labels={
            "server_id": "Server",
            "avg_latency_ms": "Average Latency (ms)"
        }
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(coloraxis_showscale=False)

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

# -------------------------------------------------
# TAB 3: DISCONNECT ANALYSIS
# -------------------------------------------------

with tab3:
    st.subheader("Disconnect Analysis")

    st.caption(
        "Disconnect rate is the percentage of log events "
        "marked DISCONNECT, not the percentage of sessions."
    )

    top_disconnects = (
        disconnects.nlargest(
            15, "disconnect_rate_pct"
        )
        .sort_values("disconnect_rate_pct")
    )

    fig = px.bar(
        top_disconnects,
        x="disconnect_rate_pct",
        y="server_id",
        orientation="h",
        color="disconnect_rate_pct",
        color_continuous_scale="Reds",
        text="disconnect_rate_pct",
        title="15 Servers with Highest Disconnect Event Rates",
        labels={
            "server_id": "Server",
            "disconnect_rate_pct": "Disconnect Rate (%)"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )
    fig.update_xaxes(
        range=[
            0,
            top_disconnects[
                "disconnect_rate_pct"
            ].max() * 1.25
        ]
    )
    fig.update_layout(
        height=550,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Disconnect rates vary only modestly across servers. "
        "This reflects the synthetic data-generation process."
    )

    st.dataframe(
        disconnects.sort_values(
            "disconnect_rate_pct",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

# -------------------------------------------------
# TAB 4: PEAK GAMING HOURS
# -------------------------------------------------

with tab4:
    st.subheader("Peak Gaming Hours")

    hourly = peak.sort_values("event_hour").copy()
    hourly["Hour"] = hourly["event_hour"].apply(
        lambda h: f"{int(h):02d}:00"
    )

    fig = px.bar(
        hourly,
        x="Hour",
        y="total_events",
        color="total_events",
        color_continuous_scale="Viridis",
        title="Game Events by Hour of Day",
        labels={
            "total_events": "Recorded Events"
        }
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        height=500,
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

    busiest = hourly.loc[
        hourly["total_events"].idxmax()
    ]
    quietest = hourly.loc[
        hourly["total_events"].idxmin()
    ]

    a, b = st.columns(2)

    a.metric(
        "Busiest Hour",
        busiest["Hour"],
        f"{int(busiest['total_events']):,} events",
        delta_color="off"
    )
    b.metric(
        "Quietest Hour",
        quietest["Hour"],
        f"{int(quietest['total_events']):,} events",
        delta_color="off"
    )

    st.caption(
        "Hourly activity is aggregated across the simulated "
        "90-day dataset. It represents event counts, not "
        "concurrent player counts."
    )

# -------------------------------------------------
# TAB 5: LOGISTIC REGRESSION
# -------------------------------------------------

with tab5:
    st.subheader("Spark MLlib Disconnect Prediction")

    models = {
        "Baseline (0.50)": (baseline, baseline_cm),
        "Adjusted (0.01)": (updated, updated_cm)
    }

    comparison = []

    for name, (metrics, cm) in models.items():
        tn, fp, fn, tp, precision, recall, f1 = (
            confusion_stats(cm)
        )

        comparison.append({
            "Model": name,
            "Accuracy": metric_value(metrics, "accuracy"),
            "ROC-AUC": metric_value(metrics, "roc_auc"),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        })

    comparison_df = pd.DataFrame(comparison)

    a, b = st.columns(2)

    with a:
        st.metric(
            "Baseline Accuracy",
            f"{comparison_df.iloc[0]['Accuracy']:.2%}"
        )
        st.metric(
            "Baseline Disconnect Recall",
            f"{comparison_df.iloc[0]['Recall']:.2%}"
        )

    with b:
        st.metric(
            "Adjusted Accuracy",
            f"{comparison_df.iloc[1]['Accuracy']:.2%}"
        )
        st.metric(
            "Adjusted Disconnect Recall",
            f"{comparison_df.iloc[1]['Recall']:.2%}"
        )

    plot_data = comparison_df.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        plot_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        title="Disconnect-Class Performance Comparison"
    )
    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )
    fig.update_yaxes(
        tickformat=".0%",
        rangemode="tozero"
    )

    st.plotly_chart(fig, use_container_width=True)

    selected = st.selectbox(
        "View Confusion Matrix",
        list(models.keys())
    )

    metrics, cm = models[selected]
    tn, fp, fn, tp, precision, recall, f1 = (
        confusion_stats(cm)
    )

    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Normal", "Actual Disconnect"],
        columns=[
            "Predicted Normal",
            "Predicted Disconnect"
        ]
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix: {selected}"
    )

    st.plotly_chart(fig, use_container_width=True)

    display_df = comparison_df.copy()

    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.2%}"
        )

    display_df["ROC-AUC"] = display_df[
        "ROC-AUC"
    ].map(lambda x: f"{x:.4f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The baseline model predicts no disconnect events. "
        "Lowering the classification threshold improves "
        "recall but creates many false positives. "
        "These models are experimental and are not "
        "suitable for production deployment."
    )

st.divider()
st.caption(
    "Distributed Game Server Analytics | "
    "Hadoop HDFS | Apache Spark SQL | Spark MLlib"
)
