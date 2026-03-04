import pandas as pd

STAGE_ORDER = [
    "landing", "browse", "product_detail",
    "add_to_cart", "checkout", "payment", "order_confirmed"
]

def build_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a funnel DataFrame with users, drop-off, and conversion at each stage."""
    stage_counts = []
    for stage in STAGE_ORDER:
        # Users who reached this stage = users who exited at this stage OR later
        idx = STAGE_ORDER.index(stage)
        reached = df[df["exited_at"].map(lambda x: STAGE_ORDER.index(x) >= idx)].shape[0]
        stage_counts.append({"stage": stage, "users": reached})
    
    funnel = pd.DataFrame(stage_counts)
    funnel["drop_off"]       = funnel["users"].shift(1) - funnel["users"]
    funnel["drop_off_pct"]   = (funnel["drop_off"] / funnel["users"].shift(1) * 100).round(2)
    funnel["stage_conv_pct"] = (funnel["users"] / funnel["users"].shift(1) * 100).round(2)
    funnel["overall_conv"]   = (funnel["users"] / funnel["users"].iloc[0] * 100).round(2)
    return funnel.fillna(0)


def top_exit_reasons(df: pd.DataFrame, stage: str) -> pd.Series:
    """Returns value counts of exit reasons for a given stage."""
    subset = df[df["exited_at"] == stage]
    return subset["exit_reason"].value_counts()


def device_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Conversion rate by device."""
    return df.groupby("device")["converted"].agg(["sum","count"]).rename(
        columns={"sum":"conversions","count":"sessions"}
    ).assign(conv_rate=lambda x: (x.conversions/x.sessions*100).round(2))


def hourly_drop_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Drop-off rate by hour of day."""
    return df.groupby("hour")["converted"].agg(
        sessions="count", conversions="sum"
    ).assign(drop_rate=lambda x: ((1 - x.conversions/x.sessions)*100).round(2))