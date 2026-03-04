import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def simulate_ab_test(df: pd.DataFrame, seed=42):
    """
    Simulate A/B test: Control = current flow, Treatment = guest checkout enabled.
    Treatment group gets a 22% relative uplift at cart → checkout stage.
    """
    np.random.seed(seed)
    df = df.copy()
    df["ab_group"] = np.random.choice(["control","treatment"], size=len(df), p=[0.5, 0.5])

    # Treatment: reduce forced_login drop-offs
    mask = (
        (df["ab_group"] == "treatment") &
        (df["exited_at"] == "add_to_cart") &
        (df["exit_reason"] == "forced_login")
    )
    # Recover ~22% of these users — they proceed to checkout
    recover_idx = df[mask].sample(frac=0.22, random_state=seed).index
    df.loc[recover_idx, "exited_at"]   = "order_confirmed"
    df.loc[recover_idx, "exit_reason"] = None
    df.loc[recover_idx, "converted"]   = 1
    df.loc[recover_idx, "order_value"] = df.loc[recover_idx, "cart_value"]

    return df


def run_hypothesis_test(ab_df: pd.DataFrame):
    """
    Two-proportion z-test + Chi-square for conversion rate difference.
    Also computes: relative uplift, confidence interval, revenue impact.
    """
    groups = ab_df.groupby("ab_group")["converted"].agg(["sum","count"]).rename(
        columns={"sum":"conversions","count":"sessions"}
    )
    groups["conv_rate"] = groups["conversions"] / groups["sessions"]

    ctrl = groups.loc["control"]
    trt  = groups.loc["treatment"]

    # Chi-square test
    ct = pd.crosstab(ab_df["ab_group"], ab_df["converted"])
    chi2, p_chi, _, _ = chi2_contingency(ct)

    # Two-proportion z-test
    p1, p2 = ctrl["conv_rate"], trt["conv_rate"]
    n1, n2 = ctrl["sessions"],  trt["sessions"]
    p_pool = (ctrl["conversions"] + trt["conversions"]) / (n1 + n2)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z      = (p2 - p1) / se
    p_val  = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% Confidence Interval for the difference
    diff   = p2 - p1
    ci_lo  = diff - 1.96 * se
    ci_hi  = diff + 1.96 * se

    # Revenue impact
    avg_order = ab_df[ab_df["converted"] == 1]["order_value"].mean()
    extra_conversions = trt["conversions"] - ctrl["conversions"]
    revenue_lift = extra_conversions * avg_order

    return {
        "groups":            groups,
        "z_stat":            round(z, 4),
        "p_value":           round(p_val, 6),
        "chi2":              round(chi2, 4),
        "p_chi":             round(p_chi, 6),
        "significant":       p_val < 0.05,
        "relative_uplift":   round((p2 - p1) / p1 * 100, 2),
        "ci_95":             (round(ci_lo, 4), round(ci_hi, 4)),
        "avg_order_value":   round(avg_order, 2),
        "extra_conversions": int(extra_conversions),
        "revenue_lift":      round(revenue_lift, 2),
    }


def plot_ab_results(result: dict, save_path=None):
    groups = result["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0e0e10")

    # Conversion rate bars
    ax = axes[0]; ax.set_facecolor("#161618")
    colors = ["#3d7fff","#ff6b35"]
    bars = ax.bar(groups.index, groups["conv_rate"] * 100, color=colors, width=0.4, edgecolor="none")
    for bar, (_, row) in zip(bars, groups.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{row['conv_rate']*100:.2f}%", ha="center", color="#ccc", fontsize=12)
    ax.set_title("Conversion Rate: Control vs Treatment", color="#f0ede8", fontsize=12)
    ax.set_ylabel("Conversion Rate (%)", color="#888")
    ax.tick_params(colors="#666")
    sig_text = f"p = {result['p_value']} → {'✅ Significant' if result['significant'] else '❌ Not Significant'}"
    ax.set_xlabel(sig_text, color="#4caf8a" if result["significant"] else "#ff3b3b", fontsize=10)

    # CI plot
    ax2 = axes[1]; ax2.set_facecolor("#161618")
    diff = result["ci_95"]
    mid  = (diff[0] + diff[1]) / 2
    ax2.barh([0], [diff[1] - diff[0]], left=diff[0], height=0.3,
             color="#ff6b35", alpha=0.7)
    ax2.axvline(0, color="#ff3b3b", linestyle="--", linewidth=1.5)
    ax2.plot(mid, 0, "o", color="#fff", zorder=5)
    ax2.set_xlim(diff[0] - 0.01, diff[1] + 0.01)
    ax2.set_yticks([]); ax2.set_xlabel("Difference in Conversion Rate", color="#888")
    ax2.set_title("95% Confidence Interval", color="#f0ede8", fontsize=12)
    ax2.tick_params(colors="#666")
    ax2.text(mid, 0.2, f"Uplift: +{result['relative_uplift']}%",
             ha="center", color="#4caf8a", fontsize=11)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()