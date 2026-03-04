import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def build_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly cohort analysis — what % of each month's users convert in later months.
    Simulates retention by treating each returning session as repeat behavior.
    """
    df = df.copy()
    df["date"]          = pd.to_datetime(df["date"])
    df["cohort_month"]  = df.groupby("session_id")["date"].transform("min").dt.to_period("M")
    df["order_month"]   = df["date"].dt.to_period("M")
    df["period_number"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

    cohort_data = df[df["converted"] == 1].groupby(
        ["cohort_month","period_number"]
    )["session_id"].nunique().reset_index()

    cohort_counts = cohort_data.pivot(
        index="cohort_month", columns="period_number", values="session_id"
    )
    cohort_size   = cohort_counts.iloc[:, 0]
    retention     = cohort_counts.divide(cohort_size, axis=0).round(3) * 100

    return retention


def compute_rfm(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """
    RFM scoring on converted sessions only.
    R = days since last purchase, F = number of purchases, M = total spend.
    """
    converted = df[df["converted"] == 1].copy()
    converted["date"] = pd.to_datetime(converted["date"])

    if reference_date is None:
        reference_date = converted["date"].max() + pd.Timedelta(days=1)

    rfm = converted.groupby("session_id").agg(
        recency  = ("date",        lambda x: (reference_date - x.max()).days),
        frequency= ("session_id",  "count"),
        monetary = ("order_value", "sum")
    ).reset_index()

    # Score 1–5 (5 = best)
    rfm["R_score"] = pd.qcut(rfm["recency"],   5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"],  5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    def segment(score):
        if score >= 13: return "Champions"
        if score >= 10: return "Loyal Customers"
        if score >= 7:  return "Potential Loyalists"
        if score >= 5:  return "At Risk"
        return "Lost"

    rfm["segment"] = rfm["RFM_score"].apply(segment)
    return rfm


def plot_cohort_heatmap(retention: pd.DataFrame, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0e0e10")
    ax.set_facecolor("#161618")
    sns.heatmap(
        retention.fillna(0), annot=True, fmt=".1f", ax=ax,
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        linewidths=0.5, linecolor="#0e0e10",
        cbar_kws={"label": "Retention %"}
    )
    ax.set_title("Monthly Cohort Retention (%)", color="#f0ede8", fontsize=13, pad=14)
    ax.set_xlabel("Months Since First Purchase", color="#888")
    ax.set_ylabel("Cohort Month", color="#888")
    ax.tick_params(colors="#aaa")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def plot_rfm_segments(rfm: pd.DataFrame, save_path=None):
    seg_counts = rfm["segment"].value_counts()
    seg_rev    = rfm.groupby("segment")["monetary"].sum().reindex(seg_counts.index)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0e0e10")
    colors = ["#ff6b35","#ffa552","#ffc87a","#4caf8a","#3d7fff"]

    # Segment size
    axes[0].set_facecolor("#161618")
    axes[0].barh(seg_counts.index, seg_counts.values, color=colors, edgecolor="none")
    for i, (val, label) in enumerate(zip(seg_counts.values, seg_counts.index)):
        axes[0].text(val + 2, i, str(val), va="center", color="#ccc", fontsize=10)
    axes[0].set_title("RFM Segment Sizes", color="#f0ede8", fontsize=12)
    axes[0].tick_params(colors="#888")

    # Revenue per segment
    axes[1].set_facecolor("#161618")
    axes[1].barh(seg_rev.index, seg_rev.values, color=colors, edgecolor="none")
    axes[1].set_title("Revenue by RFM Segment", color="#f0ede8", fontsize=12)
    axes[1].tick_params(colors="#888")
    axes[1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K")
    )

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()