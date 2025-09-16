import json
import numpy as np
import networkx as nx

RISK_BUCKETS = [(0, 40, "정상"), (40, 70, "의심"), (70, 100, "위험")]

def bucketize(score: float):
    for lo, hi, label in RISK_BUCKETS:
        if lo <= score < hi:
            return label
    return "위험"

def text_reason_from_row(row, top_feats=None):
    msgs = []
    # 규칙 점수
    if "rule_score" in row and row["rule_score"] >= 80:
        msgs.append("규칙 기반 강한 이상 징후(rule_score≥80)")
    elif "rule_score" in row and row["rule_score"] >= 50:
        msgs.append("규칙 기반 중간 수준 이상 징후(rule_score≥50)")
    # 시간/금액/외국/프록시
    if row.get("is_night", 0) == 1:
        msgs.append("심야 시간대 거래")
    if row.get("is_foreign", 0) == 1:
        msgs.append("해외(외국) 접속/거래")
    if row.get("is_proxy_ip", 0) == 1:
        msgs.append("프록시/우회 IP 사용")
    if row.get("tx_amount_avg_30d", 0) > 0 and row.get("amount", 0) > row["tx_amount_avg_30d"] * 5:
        msgs.append("30일 평균 대비 과도한 고액 송금(>5배)")
    if row.get("burst_flag", 0) == 1:
        msgs.append("단기간 다건(burst) 발생")
    if top_feats:
        msgs.append(f"모델 주요 요인: {', '.join(top_feats)}")
    return " · ".join(msgs) if msgs else "특이사항 없음"

def summarize_history(row):
    pts = []
    if row.get("tx_count_7d", 0) >= 10:
        pts.append(f"최근 7일 거래 {int(row['tx_count_7d'])}건")
    if row.get("tx_interval_avg", 0) > 0:
        pts.append(f"평균 간격 {row['tx_interval_avg']:.1f}분")
    if row.get("cp_unique_count_30d", 0) >= 10:
        pts.append(f"30일 유니크 상대 {int(row['cp_unique_count_30d'])}명")
    if row.get("tx_amount_avg_30d", 0) > 0:
        ratio = row.get("amount", 0) / (row["tx_amount_avg_30d"] + 1e-9)
        pts.append(f"현재 금액/30일 평균 = {ratio:.2f}배")
    return " / ".join(pts) if pts else "최근 거래 패턴 특이사항 없음"

def build_network_json(row):
    """
    간단 버전:
    - 노드: 송금자(account) / 수취자(counterparty) / (옵션) 거래소(exchange)
    - 엣지: account -> counterparty, counterparty -> exchange (있으면)
    """
    G = nx.DiGraph()
    acc = row.get("account_id", "acc")
    cp = row.get("counterparty_id", "cp")
    ex = row.get("exchange_id", None)

    G.add_node(acc, type="송금자")
    G.add_node(cp, type="수취자")
    G.add_edge(acc, cp, type="송금", amount=row.get("amount", 0.0))

    if ex and isinstance(ex, str) and len(ex) > 0:
        G.add_node(ex, type="거래소", conc=row.get("exchange_concentration_ratio", None))
        G.add_edge(cp, ex, type="이체/매입")

    nodes = [{"id": n, **d} for n, d in G.nodes(data=True)]
    edges = [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}

def fuse_rule_and_model(rule_score: float, model_prob: float, w_rule=0.35, w_model=0.65):
    """
    rule_score : 0~100
    model_prob : 0~1
    앙상블 스코어(0~100)
    """
    model_score = model_prob * 100.0
    return float(w_rule * rule_score + w_model * model_score)
