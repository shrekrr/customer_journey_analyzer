import pandas as pd
import numpy as np

STAGES = [
    "landing", "browse", "product_detail",
    "add_to_cart", "checkout", "payment", "order_confirmed"
]

DROP_PROBS = [0.0, 0.27, 0.28, 0.41, 0.43, 0.37, 0.37]

EXIT_REASONS = {
    "browse":          ["slow_load", "no_relevant_product", "unclear_nav"],
    "product_detail":  ["high_price", "poor_images", "no_reviews", "out_of_stock"],
    "add_to_cart":     ["forced_login", "high_shipping", "price_comparison"],
    "checkout":        ["unexpected_cost", "complex_form", "no_guest_checkout"],
    "payment":         ["payment_method_missing", "payment_error", "otp_friction"],
    "order_confirmed": []
}

def generate_sessions(n_sessions=15000, seed=42):
    np.random.seed(seed)

    # Cohort dates — simulate 6 months of data
    dates = pd.date_range("2024-01-01", "2024-06-30", periods=n_sessions)
    dates = dates.to_numpy().copy()
    np.random.shuffle(dates)

    records = []
    for i in range(n_sessions):
        device      = np.random.choice(["mobile","desktop","tablet"], p=[0.58,0.32,0.10])
        load_time   = np.random.lognormal(mean=0.8, sigma=0.5)
        hour        = np.random.randint(0, 24)
        is_weekend  = int(pd.Timestamp(dates[i]).dayofweek >= 5)
        is_new_user = int(np.random.rand() < 0.6)
        price_range = np.random.choice(["budget","mid","premium"], p=[0.4,0.4,0.2])
        n_pages_viewed = np.random.randint(1, 20)
        has_coupon  = int(np.random.rand() < 0.25)
        cart_value  = np.round(np.random.lognormal(mean=6.5, sigma=0.8), 2)
        utm_source  = np.random.choice(["organic","paid","social","email","direct"], p=[0.3,0.25,0.2,0.15,0.1])

        exited_at   = "order_confirmed"
        exit_reason = None

        for j, stage in enumerate(STAGES[1:], 1):
            prob = DROP_PROBS[j]
            if device == "mobile":      prob *= 1.20
            if load_time > 4:           prob *= 1.15
            if is_new_user:             prob *= 1.10
            if has_coupon:              prob *= 0.80
            if price_range == "premium":prob *= 1.12
            if utm_source == "email":   prob *= 0.85

            if np.random.rand() < min(prob, 0.95):
                exited_at = STAGES[j - 1]
                reasons = EXIT_REASONS.get(stage, [])
                exit_reason = np.random.choice(reasons) if reasons else None
                break

        order_value = cart_value if exited_at == "order_confirmed" else 0.0

        records.append({
            "session_id":    i,
            "date":          pd.Timestamp(dates[i]).date(),
            "month":         pd.Timestamp(dates[i]).strftime("%Y-%m"),
            "device":        device,
            "load_time_s":   round(load_time, 2),
            "hour":          hour,
            "is_weekend":    is_weekend,
            "is_new_user":   is_new_user,
            "price_range":   price_range,
            "n_pages_viewed":n_pages_viewed,
            "has_coupon":    has_coupon,
            "cart_value":    cart_value,
            "utm_source":    utm_source,
            "exited_at":     exited_at,
            "exit_reason":   exit_reason,
            "converted":     int(exited_at == "order_confirmed"),
            "order_value":   order_value,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_sessions()
    df.to_csv("data/raw/sessions.csv", index=False)
    print(df.shape)
    print(df["converted"].value_counts(normalize=True).round(3))