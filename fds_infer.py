import pandas as pd
import numpy as np
import joblib
import json
from fds_utils import (
    bucketize, fuse_rule_and_model, text_reason_from_row,
    summarize_history, build_network_json
)
from fds_pipeline import load_data, make_xy, train_model, CSV_PATH

MODEL_PATH = "model.joblib"
ENC_PATH   = "enc.joblib"
FEAT_PATH  = "features.txt"

# =====================
# 전처리 함수
# =====================
def clean_row(row_df, cat_cols, num_cols):
    """
    카테고리 값 = 빈값/NaN → 'UNKNOWN'
    수치형 값   = 빈값/NaN → 0.0
    """
    for col in cat_cols:
        if col in row_df:
            row_df[col] = (
                row_df[col]
                .astype(str)
                .replace(["", "nan", "NaN", "None"], "UNKNOWN")
                .fillna("UNKNOWN")
            )

    for col in num_cols:
        if col in row_df:
            row_df[col] = pd.to_numeric(row_df[col], errors="coerce").fillna(0.0)

    return row_df

# =====================
# 저장 & 로드
# =====================
def save_artifacts(model, enc, feat_names):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(enc,   ENC_PATH)
    with open(FEAT_PATH, "w") as f:
        f.write("\n".join(feat_names))

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    enc   = joblib.load(ENC_PATH)
    with open(FEAT_PATH) as f:
        feat_names = [l.strip() for l in f]
    return model, enc, feat_names

# =====================
# 학습 & 저장
# =====================
def fit_and_export():
    df = load_data(CSV_PATH)
    X, y, enc, feat_names = make_xy(df)
    model = train_model(X, y)
    save_artifacts(model, enc, feat_names)

# =====================
# 추론 (단일 트랜잭션)
# =====================
def infer_one(tx_row_dict, model=None, enc=None, feat_names=None):
    # model/enc가 안 들어오면 내부에서 다시 로드
    if model is None or enc is None or feat_names is None:
        model, enc, feat_names = load_artifacts()

    from fds_pipeline import CAT_COLS

    row_df = pd.DataFrame([tx_row_dict])

    NUM_COLS = [
        "amount","fee","hour","is_night","is_foreign","is_new_cp","is_proxy_ip",
        "account_age_days","tx_count_7d","tx_amount_avg_30d","tx_amount_std_30d",
        "in_degree","out_degree","is_hub_account","cp_unique_count_30d",
        "tx_interval_avg","burst_flag","day_of_week","exchange_concentration_ratio",
        "rule_score"
    ]

    # 🔧 전처리 적용
    row_df = clean_row(row_df, CAT_COLS, NUM_COLS)

    # 카테고리 인코딩
    X_cat = enc.transform(row_df[CAT_COLS])
    # 수치형
    X_num = row_df[NUM_COLS].to_numpy()
    X = np.hstack([X_num, X_cat])

    # 예측 확률 & 점수
    prob = float(model.predict_proba(X)[0,1])
    fused = fuse_rule_and_model(tx_row_dict.get("rule_score", 0), prob)
    label = bucketize(fused)

    # 설명 요소
    reason = text_reason_from_row(tx_row_dict)
    hist   = summarize_history(tx_row_dict)
    graph  = build_network_json(tx_row_dict)

    # 최종 딕셔너리 반환
    return {
        "tx_id": tx_row_dict.get("tx_id"),
        "event_time": str(tx_row_dict.get("event_time")),
        "risk": {
            "score": round(fused, 1),
            "label": label,
            "model_prob": round(prob*100, 1),
            "rule_score": tx_row_dict.get("rule_score", 0)
        },
        "explanation": {
            "reason": reason,
            "history": hist
        },
        "network": graph
    }



# =====================
# 실행 (일부 출력 + 라벨 분포 집계)
# =====================
if __name__ == "__main__":
    fit_and_export()
    df = load_data(CSV_PATH)

    model, enc, feat_names = load_artifacts()
    rows = df.to_dict(orient="records")

    # ✅ 전체 결과 추론
    results = [infer_one(row, model, enc, feat_names) for row in rows]

    # 일부만 출력
    for r in results[:5]:
        print(json.dumps(r, ensure_ascii=False, indent=2))

    # ✅ 라벨 분포 집계
    labels = pd.Series([r["risk"]["label"] for r in results]).value_counts()

    print("\n=== 라벨 분포 ===")
    for label, count in labels.items():
        print(f"{label}: {count}건")
