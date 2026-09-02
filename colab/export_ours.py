# -*- coding: utf-8 -*-
"""우리 전처리로 학습한 TabM 을 제출용 npz 로 내보낸다.

왜 새로 쓰나
    submit_14 는 지인 전처리로 만들었는데, 관문에서 우리 전처리가 시드 3개 모두
    높았다(단독 +6.7, 혼합 +1.2). 우리 쪽으로 바꾸면 제출본에서 피처 파이프라인이
    하나로 줄어든다 — 지금은 build_features(CatBoost·flatMLP용)와
    preprocess.py(TabM용)를 둘 다 돌린다. 추론 시간도 줄고 divergence 원인도 사라진다.

전처리 메타를 지인 형식(이름 기반 dict)이 아니라 우리 형식(위치 기반 배열)으로 담는다.
    지인 _tabm_transform 은 X[col].map({값: i+1}) 로 범주를 찍는데, 매핑 키의
    dtype 이 열 dtype 과 안 맞으면 전부 미지값(0)이 되고도 조용히 넘어간다.
    우리 flatMLP 경로가 쓰는 searchsorted 방식은 그 함정이 없고 이미 검증됐다.
    그래서 같은 규약으로 통일한다.

시즌가중
    sample_weight = decay ** (season - 2019). 데이터를 하나도 안 버리면서
    최근 시즌 쪽으로 학습 분포를 기울인다. 관문에서 곡선이 단조로 올라가다
    3~4 에서 포화했다 (없음 867.5 / 1.5 882.8 / 2.0 895.5 / 3.0 906.0 / 4.0 907.6).
    CatBoost 최적값은 2.0 이었는데 TabM 은 더 세게 걸어야 한다.

사용법
    python export_ours.py 2025 [ep1] [k] [decay] [ep2] [lowcard]
      ep2      Stage2 에폭 (기본 1)
      lowcard  1 이면 월/요일/이닝/볼/스트라이크/아웃을 범주형으로 추가
      load     1 이면 등판 강도 2열 추가
      regime   퓨처스 옛 체제(2019~2022) 처리.
               'flag' 플래그만, 'flag_w0.1' 플래그 + all/futures 가중 0.1,
               'flag_f0.1' 플래그 + futures 브랜치만 가중 0.1
    VS=2025 면 2019~2024 전체가 학습 구간이다(제출용).
    VS=2024 면 관문 재현용.
"""
import json
import os
import sys

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "/workspace/aimers/data"
OUT = "/workspace/aimers/out/export_ours"
SEEDS = (42, 1, 777)
OLD_F_MAX = 2022   # 이 해까지의 퓨처스가 옛 체제

import features44 as F                                          # noqa: E402


def prep_stats(X, m_tr, cat_idx):
    """G.prep 과 같은 계산을 하되, 추론 때 재현할 수 있게 통계를 함께 돌려준다."""
    n, p = X.shape
    ci = np.asarray(cat_idx)
    ni = np.asarray([j for j in range(p) if j not in set(cat_idx)])
    st = {"cat_idx": ci, "num_idx": ni}
    Xc = np.zeros((n, len(ci)), dtype=np.int64)
    cards = []
    for a, j in enumerate(ci):
        vals = np.unique(X[m_tr, j])
        vals = vals[~np.isnan(vals)]
        st[f"catkey_{a}"] = vals.astype(np.float64)
        st[f"catval_{a}"] = np.arange(1, len(vals) + 1, dtype=np.int64)
        pos = np.clip(np.searchsorted(vals, X[:, j]), 0, max(len(vals) - 1, 0))
        hit = (vals[pos] == X[:, j]) if len(vals) else np.zeros(n, bool)
        Xc[:, a] = np.where(hit, pos + 1, 0)
        cards.append(len(vals) + 1)
    Xn = X[:, ni].astype(np.float64)
    miss = np.isnan(Xn)
    med = np.nanmedian(Xn[m_tr], 0)
    Xn = np.where(miss, med, Xn)
    mu, sd = Xn[m_tr].mean(0), Xn[m_tr].std(0) + 1e-6   # 1e-6 을 sd 에 녹여 저장
    Xn = ((Xn - mu) / sd).astype(np.float32)
    hn = miss[m_tr].any(0)
    st.update(med=med, mu=mu, sd=sd, has_nan=hn)
    if hn.any():
        Xn = np.concatenate([Xn, miss[:, hn].astype(np.float32)], 1)
    return Xn, Xc, np.asarray(cards), st


def export(model, path, st, cards, n_num, cfg, features):
    """추론에 필요한 것만 배열로 뽑는다. script.py 의 predict_frame 이 읽는 형식."""
    s = model.state_dict()
    a = {"num_embedding_weight": s["num_module.linear.weight"].cpu().numpy(),
         "num_embedding_bias": s["num_module.linear.bias"].cpu().numpy()}
    for b in range(cfg["n_blocks"]):
        p = f"backbone.blocks.{b}.0."
        for key in ("weight", "r", "s", "bias"):
            a[f"block_{b}_{key}"] = s[p + key].cpu().numpy()
    a["output_weight"] = s["output.weight"].cpu().numpy()
    a["output_bias"] = s["output.bias"].cpu().numpy()
    a.update({k: np.asarray(v) for k, v in st.items()})
    a["meta"] = np.asarray(json.dumps(
        {"features": list(features), "cards": [int(c) for c in cards],
         "n_num": int(n_num), **cfg}, ensure_ascii=False))
    np.savez_compressed(path, **a)


if __name__ == "__main__":
    VS = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    EP1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    DECAY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # 0 이면 가중 없음
    EP2 = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    LOWCARD = len(sys.argv) > 6 and sys.argv[6] == "1"
    LOAD = len(sys.argv) > 7 and sys.argv[7] == "1"
    REGIME = sys.argv[8] if len(sys.argv) > 8 else ""
    os.makedirs(OUT, exist_ok=True)

    from rtdl_num_embeddings import LinearReLUEmbeddings
    from tabm import TabM
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG = dict(k=K, n_blocks=3, d_block=256, dropout=0.1, d_emb=16,
               num_embeddings="linear_relu", backbone_kind="batch_ensemble",
               ep1=EP1, ep2=EP2, lr1=2e-3, lr2=2e-4, vs=VS, decay=DECAY,
               lowcard=LOWCARD, load=LOAD, regime=REGIME)

    d = F.build(DATA, VS=VS)
    X, y, m_tr = d["X44"], d["y"], d["m_tr"]
    season, is_f = d["season"], d["is_f"]
    F44 = list(d["F44"])
    if LOAD:
        # pn_cur 과 game_month 는 둘 다 44열에 있는데 그 '비율' 은 모델이 스스로
        # 만들어야 한다. 나눗셈은 신경망이 잘 못 배우는 형태라 명시적으로 준다.
        # 8시드 짝비교에서 혼합 +2.4, 8/8 양수, t=3.41 로 확정했다.
        pn = X[:, F44.index("pn_cur")].astype(np.float64)
        mo = X[:, F44.index("game_month")].astype(np.float64)
        el = np.clip(mo - 2.0, 1.0, None)              # 3월 개막
        X = np.concatenate([X, np.stack(
            [np.log1p(pn) - np.log1p(el), pn / el], 1).astype(np.float32)], 1)
        F44 = F44 + ["load_rate", "load_lin"]
    # 퓨처스 체제 플래그. 2023 년에 퓨처스 성공률이 70.9% -> 47.3% 로 무너진다
    # (같은 해 1군은 -0.06%p). 학습 퓨처스의 80.4% 가 옛 체제라 그대로 섞어
    # 학습하면 다른 게임을 배운다. 2025 행은 전부 새 체제(1)다.
    if REGIME:
        newreg = ((is_f & (season > OLD_F_MAX)) | (~is_f)).astype(np.float32)
        X = np.concatenate([X, newreg[:, None]], 1)
        F44 = F44 + ["abs_regime"]
    cat_idx = list(d["cat_idx"])
    if REGIME:
        cat_idx = cat_idx + [F44.index("abs_regime")]
    if LOWCARD:
        # 이닝 1회와 2회의 차이가 선형이라는 가정을 깬다. 카디널리티가 작아
        # 범주로 넣어도 파라미터가 거의 안 늘어난다.
        cat_idx = sorted(set(cat_idx) | {F44.index(c) for c in
            ("game_month", "game_dayofweek", "inning",
             "balls_before", "strikes_before", "outs_before")})
    Xn, Xc, cards, st = prep_stats(X, m_tr, cat_idx)
    print(f"VS={VS} ep1={EP1} ep2={EP2} k={K} decay={DECAY} lowcard={LOWCARD} load={LOAD} regime={REGIME!r}  학습 {m_tr.sum():,}행  "
          f"수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열", flush=True)

    import tabm_gate_gpu as G
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    G.YY = torch.from_numpy(y.astype(np.float32))
    G.Xn, G.cards = Xn, cards

    tr_idx = np.where(m_tr)[0]
    # 시즌가중. Stage2 는 마지막 시즌만 쓰므로 그 안에서는 가중이 상수라 영향이 없다.
    W = None if not DECAY else DECAY ** (season[tr_idx].astype(np.float64) - 2019)
    # 옛 체제 퓨처스에 걸 가중. 브랜치마다 다르게 걸 수 있다.
    OLD_F = is_f[tr_idx] & (season[tr_idx] <= OLD_F_MAX)
    RW = {"flag": {}, "flag_w0.1": {"all": 0.1, "futures": 0.1, "regular": 0.1},
          "flag_f0.1": {"futures": 0.1}}.get(REGIME, {})
    for seed in SEEDS:
        # regular 브랜치를 뺐다. 라우팅 alpha 스윕에서 1군 행에 regular 를 섞는 게
        # 손해였다(aR=0 이 최적, 행 부트스트랩 200회 중 196회). all 의 89%가 이미
        # regular 데이터라 같은 걸 배운다. 빼면 추론 패스가 절반 가까이 줄어
        # 같은 시간에 시드를 3개 넣을 수 있다.
        for br, sel in (("all", np.ones(len(tr_idx), bool)),
                        ("futures", is_f[tr_idx])):
            idx = tr_idx[sel]
            w = None if W is None else W[sel].copy()
            ow = RW.get(br, 1.0)
            if ow != 1.0:
                w = (np.ones(int(sel.sum())) if w is None else w)
                w = w * np.where(OLD_F[sel], ow, 1.0)
            torch.manual_seed(seed)             # 초기화까지 덮는다
            torch.cuda.manual_seed_all(seed)
            m = TabM.make(
                n_num_features=Xn.shape[1],
                cat_cardinalities=[int(c) for c in cards], d_out=1,
                num_embeddings=LinearReLUEmbeddings(Xn.shape[1], d_embedding=16),
                arch_type="tabm", k=K, n_blocks=3, d_block=256, dropout=0.1,
            ).to(DEV)
            G.train(m, idx, EP1, 2e-3, w=w, seed=seed, tag=f"s{seed} {br} S1")
            s2 = season[idx] == VS - 1
            G.train(m, idx[s2], EP2, 2e-4, params=G.stage2_params(m), seed=seed,
                    tag=f"s{seed} {br} S2")
            export(m, os.path.join(OUT, f"{br}_seed{seed}.npz"),
                   st, cards, Xn.shape[1], CFG, F44)
            print(f"  저장 {br}_seed{seed}.npz", flush=True)
            del m
            torch.cuda.empty_cache()
    print("완료")
