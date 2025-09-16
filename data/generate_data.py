import pandas as pd
import numpy as np

# ----------------------------
# Config
# ----------------------------
N = 80_000
N_ACCOUNTS = 6000
N_COUNTERPARTIES = 7500
N_DEVICES = 8000
N_IPS = 9000

START = pd.Timestamp('2025-07-01 00:00:00')
END   = pd.Timestamp('2025-08-31 23:59:00')
DURATION_MIN = int((END - START).total_seconds() // 60)

def generate_data(seed=42):
    rng = np.random.default_rng(seed)

    # 랜덤 풀
    accounts = [f"A{str(i).zfill(5)}" for i in range(N_ACCOUNTS)]
    counterparties = [f"C{str(i).zfill(5)}" for i in range(N_COUNTERPARTIES)]
    devices = [f"D{str(i).zfill(5)}" for i in range(N_DEVICES)]
    ips = [f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,255)}" for _ in range(N_IPS)]
    
    # ----------------------------
    # 정상 거래 (60%)
    # ----------------------------
    benign_size = int(N * 0.6)
    benign_data = {
        "tx_id": range(1, benign_size+1),
        "event_time": [START + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, DURATION_MIN, size=benign_size)],
        "account_id": rng.choice(accounts, size=benign_size),
        "counterparty_id": rng.choice(counterparties, size=benign_size),
        "device_id": rng.choice(devices, size=benign_size),
        "ip": rng.choice(ips, size=benign_size),
        "country": rng.choice(["KR","US","CN","JP","SG"], size=benign_size, p=[0.8,0.05,0.05,0.05,0.05]),
        "channel": rng.choice(["app","web","atm"], size=benign_size, p=[0.6,0.35,0.05]),
        "amount": rng.normal(500, 200, size=benign_size).clip(1, 5_000_000),
        "fee": rng.normal(30, 10, size=benign_size).clip(0, 100),
        "hour": rng.integers(0, 24, size=benign_size),
        "fraud_type": "benign"
    }
    df = pd.DataFrame(benign_data)

    # ----------------------------
    # 사기 거래 (40% → 4종류 균등 분포)
    # ----------------------------
    fraud_size = N - benign_size
    fraud_types = ["smurfing", "multi_account", "proxy_boost", "night_high_amount_burst"]
    fraud_counts = {ft: fraud_size//len(fraud_types) for ft in fraud_types}

    fraud_rows = []
    tx_id_counter = benign_size + 1

    # 1) Smurfing
    for _ in range(fraud_counts["smurfing"]):
        fraud_rows.append({
            "tx_id": tx_id_counter,
            "event_time": START + pd.Timedelta(minutes=int(rng.integers(0, DURATION_MIN))),
            "account_id": rng.choice(accounts),
            "counterparty_id": rng.choice(counterparties),
            "device_id": rng.choice(devices),
            "ip": rng.choice(ips),
            "country": "KR",
            "channel": "app",
            "amount": rng.normal(100, 30),  # 소액 다건
            "fee": rng.normal(5, 2),
            "hour": rng.integers(0, 24),
            "fraud_type": "smurfing"
        })
        tx_id_counter += 1

    # 2) Multi-account
    cp_target = rng.choice(counterparties)
    for _ in range(fraud_counts["multi_account"]):
        fraud_rows.append({
            "tx_id": tx_id_counter,
            "event_time": START + pd.Timedelta(minutes=int(rng.integers(0, DURATION_MIN))),
            "account_id": rng.choice(accounts),
            "counterparty_id": cp_target,
            "device_id": rng.choice(devices),
            "ip": rng.choice(ips),
            "country": "KR",
            "channel": rng.choice(["app","web"]),
            "amount": rng.normal(2000, 500),
            "fee": rng.normal(20, 5),
            "hour": rng.integers(0, 24),
            "fraud_type": "multi_account"
        })
        tx_id_counter += 1

    # 3) Proxy boost
    for _ in range(fraud_counts["proxy_boost"]):
        fraud_rows.append({
            "tx_id": tx_id_counter,
            "event_time": START + pd.Timedelta(minutes=int(rng.integers(0, DURATION_MIN))),
            "account_id": rng.choice(accounts),
            "counterparty_id": rng.choice(counterparties),
            "device_id": rng.choice(devices),
            "ip": rng.choice(ips),
            "country": rng.choice(["US","CN","JP","SG"]),
            "channel": "web",
            "amount": rng.normal(5_000_000, 1_000_000),  # 초고액
            "fee": rng.normal(50, 15),
            "hour": rng.integers(0, 24),
            "fraud_type": "proxy_boost"
        })
        tx_id_counter += 1

    # 4) Night high-amount burst
    for _ in range(fraud_counts["night_high_amount_burst"]):
        fraud_rows.append({
            "tx_id": tx_id_counter,
            "event_time": START + pd.Timedelta(minutes=int(rng.integers(0, DURATION_MIN))),
            "account_id": rng.choice(accounts),
            "counterparty_id": rng.choice(counterparties),
            "device_id": rng.choice(devices),
            "ip": rng.choice(ips),
            "country": "KR",
            "channel": "app",
            "amount": rng.normal(3_000_000, 500_000),  # 야간 고액
            "fee": rng.normal(40, 10),
            "hour": rng.choice([0,1,2,3,4,5,23]),
            "fraud_type": "night_high_amount_burst"
        })
        tx_id_counter += 1

    fraud_df = pd.DataFrame(fraud_rows)
    df = pd.concat([df, fraud_df], ignore_index=True)

    # ----------------------------
    # 공통 Feature Engineering
    # ----------------------------
    df["is_night"] = df["hour"].isin([0,1,2,3,4,5,23]).astype(int)
    df["is_foreign"] = (df["country"] != "KR").astype(int)
    df["is_new_cp"] = rng.choice([0,1], size=len(df), p=[0.9,0.1])
    df["is_proxy_ip"] = (df["fraud_type"]=="proxy_boost").astype(int)

    df["label_fraud"] = (df["fraud_type"]!="benign").astype(int)

    df["account_age_days"] = rng.integers(30, 2000, size=len(df))
    df["tx_count_7d"] = rng.integers(0, 50, size=len(df))
    df["tx_amount_avg_30d"] = rng.normal(500_000, 200_000, size=len(df)).clip(0, 1e7)
    df["tx_amount_std_30d"] = rng.normal(100_000, 50_000, size=len(df)).clip(0, 1e6)

    df["in_degree"] = rng.integers(0, 50, size=len(df))
    df["out_degree"] = rng.integers(0, 50, size=len(df))
    df["is_hub_account"] = (df["out_degree"] > 30).astype(int)
    df["cp_unique_count_30d"] = rng.integers(1, 100, size=len(df))
    df["tx_interval_avg"] = rng.normal(300, 120, size=len(df)).clip(1, 10_000)
    df["burst_flag"] = (df["fraud_type"]=="night_high_amount_burst").astype(int)
    df["day_of_week"] = pd.to_datetime(df["event_time"]).dt.dayofweek

    df["exchange_id"] = rng.choice(["upbit","bithumb","coinone","korbit",""], size=len(df), p=[0.25,0.25,0.25,0.15,0.10])
    df["exchange_concentration_ratio"] = rng.uniform(0, 1, size=len(df)).round(2)

    df["rule_score"] = (
        50*df["is_night"] +
        50*df["is_foreign"] +
        100*df["is_proxy_ip"] +
        0.0001*df["amount"]
    ).astype(int).clip(0,200)

    df.to_csv("synthetic_transactions.csv", index=False)
    print(f"✅ synthetic_transactions.csv 생성 완료 ({len(df)} rows)")

if __name__ == "__main__":
    generate_data()
