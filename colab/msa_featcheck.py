# -*- coding: utf-8 -*-
"""MS 감사판의 추론측 58열 featurization 을 만들고, 학습측 행렬과 열 단위 대조.

이 프로젝트에서 점수를 제일 많이 깎은 결함이 학습/추론 행렬 불일치였다 (c4 사고).
MS 는 새 featurization 이 13열이라 이름이 아니라 값으로 대조한다.

대조 대상 (2024 행 20,000개)
    학습측   features44.build(VS=2025) X44 + 감사 extras + c4 + hand_pair
    추론측   패키지 preprocess Xf -> features44 이름순 재배열 -> plat_dev 덮어쓰기
             -> extras 재계산 -> 같은 순서로 조립
    plat_dev 는 따로 본다 — 학습측(2024행)은 2023까지, 추론표는 2024까지라
    2024 행에서는 다른 게 **정상**이다 (2025 행에서는 같은 정의가 된다).

msa_featurize() 는 그대로 submit_46 의 script.py 에 들어간다.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "colab"))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "open (1)", "data")
N = 20000


# ================= 추론측 (submit_46 에 그대로 들어갈 함수) =================
def msa_featurize(Xf, ordered, feat_names, plat_z, cm_z=None):
    """패키지 Xf(44열) + 원자료 ordered 에서 MS 감사판 58열을 만든다.

    feat_names: 학습이 저장한 열 순서 (44 + extras 12 + abs_regime + hand_pair)
    plat_z:     ms_plat_prior.npz (features44 규약의 2024까지 플래툰 편차)
    전부 그 행 안의 값만 쓴다. 다른 평가 행을 참조하지 않는다.
    """
    X44 = Xf[feat_names[:44]].to_numpy(dtype=np.float64).copy()
    # plat_dev 를 features44 규약(2024까지)으로 덮어쓴다
    j_plat = feat_names.index("plat_dev")
    pids = np.asarray(plat_z["pitcher_id"], np.int64)
    dev = np.asarray(plat_z["dev"], np.float64)
    pid = ordered["pitcher_id"].to_numpy(np.int64)
    hand = ordered["batter_hand"].to_numpy(np.int64)
    if len(pids):
        pos = np.searchsorted(pids, pid)
        safe = np.minimum(pos, len(pids) - 1)
        hit = (pos < len(pids)) & (pids[safe] == pid)
        X44[:, j_plat] = np.where(hit, np.where(hand == 1, dev[safe, 0],
                                                dev[safe, 1]), 0.0)
    # cm 3열도 features44 규약으로 덮어쓴다 (48칸 표 + 전역평균, 행 내부 연산)
    if cm_z is not None:
        cm = ((ordered["balls_before"].to_numpy(np.int64) * 3
               + ordered["strikes_before"].to_numpy(np.int64)) * 4
              + ordered["pitcher_hand"].to_numpy(np.int64) * 2
              + ordered["batter_hand"].to_numpy(np.int64))
        ck = np.asarray(cm_z["cm"], np.int64)
        cpos = np.searchsorted(ck, cm)
        csafe = np.minimum(cpos, len(ck) - 1)
        chit = (cpos < len(ck)) & (ck[csafe] == cm)
        lg = np.where(chit, np.asarray(cm_z["lg"], np.float64)[csafe], 0.0)
        rel = np.where(chit, np.asarray(cm_z["rel"], np.float64)[csafe], 1.0)
        gm = float(cm_z["gm"])
        X44[:, feat_names.index("lg_cm_eff")] = lg
        X44[:, feat_names.index("cm_rel")] = rel
        X44[:, feat_names.index("p_adj_cm")] =             gm + (X44[:, feat_names.index("p_is_succ")] - gm) * rel
    # 감사 extras — 학습(multistate_auditfeat.audit_features)과 같은 식
    p_n = ordered["asof_pitcher_n"].fillna(0.0).to_numpy(np.float64)
    b_n = ordered["asof_batter_n"].fillna(0.0).to_numpy(np.float64)
    mix_n = ordered["asof_pitcher_pitchmix_n"].fillna(0.0).to_numpy(np.float64)
    p_rate = ordered["asof_pitcher_success_rate"].fillna(.5).to_numpy(np.float64)
    b_rate = ordered["asof_batter_success_rate"].fillna(.5).to_numpy(np.float64)
    li = ordered["li"].fillna(1.0).to_numpy(np.float64)
    balls = ordered["balls_before"].to_numpy(np.float64)
    strikes = ordered["strikes_before"].to_numpy(np.float64)
    count12 = balls * 3.0 + strikes
    runners = ordered["base_state"].astype(str).str.count(r"[^_]") \
        .to_numpy(np.float64)
    p_is = X44[:, feat_names.index("p_is_succ")]
    hp = ordered["pitcher_hand"].fillna(0).to_numpy(np.int64)
    hb = ordered["batter_hand"].fillna(0).to_numpy(np.int64)
    hand_pair = np.where((hp >= 1) & (hp <= 2) & (hb >= 1) & (hb <= 2),
                         (hp - 1) * 2 + (hb - 1), -1).astype(np.float64)
    lp, lb, lm = np.log1p(p_n), np.log1p(b_n), np.log1p(mix_n)
    extras = {
        "log_p_n": lp, "log_b_n": lb, "log_mix_n": lm,
        "p_rate_x_log_pn": p_rate * lp, "b_rate_x_log_bn": b_rate * lb,
        "li_x_count12": li * count12, "li_x_runners": li * runners,
        "li_x_p_is": li * p_is,
        "li_x_hand00": li * (hand_pair == 0), "li_x_hand01": li * (hand_pair == 1),
        "li_x_hand10": li * (hand_pair == 2), "li_x_hand11": li * (hand_pair == 3),
    }
    isf = ordered["game_type"].astype(str).to_numpy() == "F"
    cols = [X44]
    for nm in feat_names[44:]:
        if nm == "abs_regime":
            cols.append(np.where(isf, 2.0, 3.0)[:, None])
        elif nm == "hand_pair":
            cols.append(hand_pair[:, None])
        else:
            cols.append(extras[nm][:, None])
    return np.concatenate(cols, axis=1)


# ================= 학습측 재구성 + 대조 =================
if __name__ == "__main__":
    import features44 as F
    from train_chan_3 import preprocess as PP

    z0 = np.load(os.path.join(ROOT, "msa", "msa_np_s42.npz"), allow_pickle=False)
    feat_names = [str(c) for c in json.loads(str(z0["meta"].item()))["features"]]
    print(f"  학습 열 {len(feat_names)}개  마지막 4: {feat_names[-4:]}")

    raw = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                        encoding="utf-8-sig"))
    built = F.build(DATA, VS=2025, return_frame=True)
    X44 = built["X44"].astype(np.float64)
    names = list(built["F44"])
    assert names == feat_names[:44], "features44 열 순서가 저장본과 다르다"
    # features44 frame 은 train.csv 원 순서, raw 는 row_id 정렬 — 위치 대응표
    rid_f = built["frame"]["row_id"].to_numpy()
    pos = pd.Series(np.arange(len(raw)), index=raw["row_id"].to_numpy())
    f2raw = pos.reindex(rid_f).to_numpy()      # features44 행 -> raw 행

    # 학습측 extras (audit_features 를 raw 기준으로 인라인)
    season_r = raw["season"].to_numpy()
    isf_r = raw["game_type"].astype(str).to_numpy() == "F"
    X44_r = np.empty_like(X44)
    X44_r[f2raw] = X44                          # raw 순서로 배열
    ordered_tr = raw
    TR = msa_featurize(
        pd.DataFrame(X44_r, columns=names), ordered_tr, feat_names,
        {"pitcher_id": np.zeros(0, np.int64), "dev": np.zeros((0, 2))})
    # 학습측 plat_dev 는 features44 원값 그대로 되돌린다 (빈 표는 0 을 넣으므로)
    TR[:, names.index("plat_dev")] = X44_r[:, names.index("plat_dev")]
    # 학습측 c4: old 체제 포함 4단계
    old_r = season_r <= 2022
    TR[:, feat_names.index("abs_regime")] = np.where(
        old_r & isf_r, 0.0, np.where(old_r & ~isf_r, 1.0,
                                     np.where(isf_r, 2.0, 3.0)))

    # ---- 추론측: 2024 행 20,000개를 test 형식으로
    cols_t = list(pd.read_csv(os.path.join(DATA, "test.csv"),
                              encoding="utf-8-sig", nrows=0).columns)
    for need in ("asof_pitcher_pitchmix_n", "asof_batter_n", "li"):
        assert need in cols_t, f"test.csv 에 {need} 가 없다"
    pool_idx = np.where(season_r == 2024)[0]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(pool_idx, N, replace=False))
    sub = raw.iloc[pick][cols_t].reset_index(drop=True)

    os.chdir(os.path.join(ROOT, "submit_44"))
    sys.path.insert(0, os.getcwd())
    import preprocess as PPF
    # history 를 <2024 로 적합한다. 2024 행 검사에서 "그 행 이전 전부" 관계를
    # 재현하기 위한 것 (col_audit 때 확립한 교정 프로토콜 — 제출본의
    # history.json(2024 말 기준)을 쓰면 이력창 열들이 허위 차이를 낸다).
    hist = PP.fit_history_tables(raw[raw.season < 2024])
    ordered = PPF.sort_by_row_id(sub)
    Xf = PPF.build_inference_features(ordered, hist)
    os.chdir(ROOT)
    assert set(names) == set(Xf.columns), "이름 집합 불일치"
    plat_z = np.load("C:/tmp/ms2/model/ms_plat_prior.npz", allow_pickle=False)
    cm_z = np.load(os.path.join(ROOT, "msa", "msa_cm48.npz"), allow_pickle=False)
    INF = msa_featurize(Xf, ordered, feat_names, plat_z, cm_z)

    # ---- 대조 (ordered 순서 -> raw pick 순서 대응)
    omap = pd.Series(np.arange(len(ordered)),
                     index=ordered["row_id"].to_numpy())
    o_of_pick = omap.reindex(raw.iloc[pick]["row_id"].to_numpy()).to_numpy()
    A = TR[pick]
    B = INF[o_of_pick]
    print(f"\n  {'열':26s} {'최대차':>12s} {'다른행':>8s}")
    bad = 0
    for j, nm in enumerate(feat_names):
        a, b = A[:, j], B[:, j]
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
        mx, nd = float(d.max()), int((d > 1e-6).sum())
        if nm == "plat_dev":
            print(f"  {nm:26s} {mx:12.4e} {nd:8,}   (2024행이라 다른 게 정상 — "
                  f"상관 {np.corrcoef(a, b)[0, 1]:.4f})")
            continue
        if nd:
            bad += 1
            print(f"  {nm:26s} {mx:12.4e} {nd:8,}   **불일치**")
    print(f"\n  plat_dev 제외 {len(feat_names)-1}열 중 불일치 {bad}개"
          + ("  -> 통과" if bad == 0 else "  -> **고쳐야 한다**"))
