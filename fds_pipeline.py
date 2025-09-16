import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import shap
from fds_utils import bucketize, text_reason_from_row, summarize_history, build_network_json, fuse_rule_and_model

CSV_PATH = "data/synthetic_transactions.csv"
TARGET = "label_fraud"   # 0/1
CAT_COLS = ["country", "channel", "exchange_id"]
ID_COLS  = ["tx_id","event_time","account_id","counterparty_id","device_id","ip"]
USE_COLS = [
    # 수치/이진
    "amount","fee","hour","is_night","is_foreign","is_new_cp","is_proxy_ip",
    "account_age_days","tx_count_7d","tx_amount_avg_30d","tx_amount_std_30d",
    "in_degree","out_degree","is_hub_account","cp_unique_count_30d",
    "tx_interval_avg","burst_flag","day_of_week","exchange_concentration_ratio",
    "rule_score"
] + CAT_COLS + ID_COLS + [TARGET, "fraud_type"]  # fraud_type은 분석용

def load_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    # 보장: event_time 파싱
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"])
    # 누락 컬럼 방지
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNK"
    return df

def make_xy(df):
    # 카테고리 인코딩(OrdinalEncoder; 추론 시에도 같은 카테고리 맵 유지 필요)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc.fit_transform(df[CAT_COLS])
    X_num = df[[
        "amount","fee","hour","is_night","is_foreign","is_new_cp","is_proxy_ip",
        "account_age_days","tx_count_7d","tx_amount_avg_30d","tx_amount_std_30d",
        "in_degree","out_degree","is_hub_account","cp_unique_count_30d",
        "tx_interval_avg","burst_flag","day_of_week","exchange_concentration_ratio",
        "rule_score"
    ]].fillna(0.0).to_numpy()

    X = np.hstack([X_num, X_cat])
    y = df[TARGET].astype(int).to_numpy()

    feature_names = [
        "amount","fee","hour","is_night","is_foreign","is_new_cp","is_proxy_ip",
        "account_age_days","tx_count_7d","tx_amount_avg_30d","tx_amount_std_30d",
        "in_degree","out_degree","is_hub_account","cp_unique_count_30d",
        "tx_interval_avg","burst_flag","day_of_week","exchange_concentration_ratio",
        "rule_score"
    ] + CAT_COLS
    return X, y, enc, feature_names

def train_model(X_train, y_train):
    base = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1
    )
    # 확률 보정(Platt/Isotonic 자동 선택)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)
    return model

def find_best_threshold(y_true, prob, beta=1.0):
    # F1 최적 threshold
    ths = np.linspace(0.05, 0.95, 181)
    best = (0.5, -1.0)
    for t in ths:
        pred = (prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best[1]:
            best = (t, f1)
    return best

def main():
    df = load_data(CSV_PATH)
    # 시간 기반 분할(데모: 랜덤 스플릿)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])

    X_train, y_train, enc, feat_names = make_xy(train_df)
    X_test,  y_test,  _,   _         = make_xy(test_df)

    model = train_model(X_train, y_train)
    prob_test = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, prob_test)
    th, best_f1 = find_best_threshold(y_test, prob_test)
    pred_test = (prob_test >= th).astype(int)

    print(f"[Eval] AUC={auc:.4f} | F1*={best_f1:.4f} @ thr={th:.2f}")

    # === 샘플 1건을 UI JSON으로 변환 ===
    i = 0
    row = test_df.iloc[i].to_dict()
    # 모델 확률
    p = float(model.predict_proba(X_test[[i]])[0,1])
    # 규칙+모델 앙상블
    fused = fuse_rule_and_model(row.get("rule_score", 0.0), p)
    risk_label = bucketize(fused)

    # SHAP 상위 피처명 (설명)
    try:
        # CalibratedClassifierCV 내부 추출
        xgb_est = model.base_estimator
        xgb_est.fit(X_train, y_train)   # 트리구조 필요 시 재학습(가벼움)
        import shap
        explainer = shap.TreeExplainer(xgb_est)
        shap_vals = explainer.shap_values(X_test[[i]])  # (n_features,)
        idx_top = np.argsort(np.abs(shap_vals))[::-1][:3]
        top_feats = [feat_names[k] for k in idx_top]
    except Exception:
        top_feats = None

    reason = text_reason_from_row(row, top_feats=top_feats)
    hist   = summarize_history(row)
    graph  = build_network_json(row)

    ui_json = {
        "tx_id": row.get("tx_id"),
        "event_time": str(row.get("event_time")),
        "risk_score": round(fused, 1),
        "risk_label": risk_label,
        "model_prob": round(p*100, 1),
        "rule_score": row.get("rule_score", 0.0),
        "reason": reason,
        "history_summary": hist,
        "network": graph
    }

    print("\n=== UI JSON SAMPLE ===")
    print(pd.Series(ui_json).to_json(force_ascii=False, indent=2))

if __name__ == "__main__":
    main()
