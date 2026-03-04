import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "#0e0e10",
    "axes.facecolor":   "#161618",
    "text.color":       "#e0ddd8",
    "axes.labelcolor":  "#e0ddd8",
    "xtick.color":      "#888",
    "ytick.color":      "#888",
    "grid.color":       "#2a2a2a",
    "axes.edgecolor":   "#2a2a2a",
})

ORANGE = "#ff6b35"
PALETTE = ["#ff6b35","#ffa552","#ffc87a","#ffe0aa","#4caf8a","#3d7fff","#cc88ff"]


def plot_funnel_bars(funnel: pd.DataFrame, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(funnel["stage"], funnel["users"], color=ORANGE, alpha=0.85, edgecolor="none", width=0.6)
    for bar, row in zip(bars, funnel.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f"{int(row.users):,}", ha="center", fontsize=9, color="#ccc")
        if row.drop_off_pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f"▼{row.drop_off_pct}%", ha="center", fontsize=8, color="#fff", alpha=0.7)
    ax.set_title("User Journey Funnel", fontsize=14, color="#f0ede8", pad=16)
    ax.set_ylabel("Users")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def plot_dropoff_reasons(df, stage: str, save_path=None):
    reasons = df[df["exited_at"] == stage]["exit_reason"].dropna()
    if reasons.empty:
        print(f"No exit reason data for stage: {stage}")
        return
    counts = reasons.value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=PALETTE[:len(counts)], edgecolor="none")
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=10)
    ax.set_title(f"Exit Reasons at: {stage}", fontsize=13, color="#f0ede8")
    ax.set_xlabel("Users")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def plot_device_conversion(device_df: pd.DataFrame, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [ORANGE, "#3d7fff", "#ffa552"]
    ax.bar(device_df.index, device_df["conv_rate"], color=colors, edgecolor="none", width=0.5)
    for i, (idx, row) in enumerate(device_df.iterrows()):
        ax.text(i, row.conv_rate + 0.2, f"{row.conv_rate}%", ha="center", fontsize=11, color="#ccc")
    ax.set_title("Conversion Rate by Device", fontsize=13, color="#f0ede8")
    ax.set_ylabel("Conversion Rate (%)")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def plot_plotly_funnel(funnel: pd.DataFrame):
    fig = go.Figure(go.Funnel(
        y=funnel["stage"],
        x=funnel["users"],
        textinfo="value+percent initial",
        marker=dict(color=[ORANGE]*len(funnel)),
        connector=dict(line=dict(color="#333", width=1))
    ))
    fig.update_layout(
        title="Interactive Conversion Funnel",
        paper_bgcolor="#0e0e10", plot_bgcolor="#161618",
        font=dict(color="#e0ddd8"),
        margin=dict(l=200)
    )
    fig.show()