# -*- coding: utf-8 -*-
"""CTR 계열 네 구조를 같은 조건에서 잰다 — DCNv2 / FiBiNET-SE / DIN / BST.

왜
    DeepFM(2017) 단독 856 이 ResNet(812.8) / FT-Transformer(769.6) 보다 위였다.
    "MLP 계열은 여기서 안 된다" 는 일반화에 반례가 하나 생긴 것이다.
    명시적 특징 교차가 실제로 뭔가를 한다면 더 나은 교차 구조를 볼 값어치가 있다.

판정선 (ftt_gate.py 가 세워둔 것)
    단독 880 아래면 기대하지 않는다. 실측으로 확인됐다 —
        MNCA 897.5 -> w=0.10 에서 +5.1        (880 위, 유일하게 벌어준다)
        LightGBM 837.5 -> +1.4   HistGB 831.3 -> +0.4
        ResNet 812.8 -> +0.3     FTT 769.6 -> -3.0
    그래서 단독 점수와 현행과의 상관, 소량 혼합 이득 셋을 같이 낸다.

네 구조
    DCNv2    x_{l+1} = x0 * (U V^T x_l + b) + x_l    저계수 명시적 교차
    SENet    필드별 중요도를 squeeze-excitation 으로 학습 후 재가중 (FiBiNET)
    DIN      투수 이력 시퀀스에 현재 상황으로 어텐션 -> 가중 합
    BST      같은 시퀀스에 트랜스포머 인코더

규칙 4 — DIN/BST 의 시퀀스
    2025 행의 '행동 시퀀스' 를 그 행 앞의 2025 투구로 만들면 **다른 평가 행을
    쓰는 것**이라 위반이다. 그래서 시퀀스를 (투수, 시즌) 단위로 **직전 시즌까지의
    마지막 K개 투구**로 고정한다. 그 행만 보고 조회하는 표가 된다.
    시퀀스 자체는 (투수,시즌) 상수지만 **어텐션 질의가 그 행의 카운트·타자손**이라
    출력은 행마다 다르다. 투수당 상수가 아니다 (그건 id 임베딩에 이미 흡수된다).

용량
    신호가 0.93% 라 전부 작게 잡는다. 오늘 잰 것만 해도 d_block 스윕 네 구성이
    전부 기준선 이하였고 XGBoost d6 은 -2508 이었다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
DATA = os.environ.get("AIMERS_DATA", os.path.join(ROOT, "open (1)", "data"))
DL = os.path.join(SC, "_dl")
VS = int(os.environ.get("CZ_VS", "2024"))
SEEDS = tuple(int(x) for x in os.environ.get("CZ_SEEDS", "42,1").split(","))
MODELS = tuple(x.strip() for x in os.environ.get(
    "CZ_MODELS", "DCNv2,SENet,DIN,BST"
).split(",") if x.strip())
EPOCHS, LR, BS, WD = 2, 3e-3, 2048, 3e-4
KSEQ = 16
EMB = 16
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() - 2))

import features44 as F                                          # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


# ------------------------------------------------------------------ 모델들
class DCNv2(nn.Module):
    """저계수 cross layer 3개 + 얕은 deep. 병렬 결합."""
    def __init__(self, n_num, cards, d=128, n_cross=3, rank=32):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, EMB) for c in cards])
        for e in self.emb:
            nn.init.normal_(e.weight, 0.0, 0.02)
        d0 = n_num + len(cards) * EMB
        self.U = nn.ParameterList([nn.Parameter(torch.randn(d0, rank) * 0.02)
                                   for _ in range(n_cross)])
        self.V = nn.ParameterList([nn.Parameter(torch.randn(rank, d0) * 0.02)
                                   for _ in range(n_cross)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(d0))
                                   for _ in range(n_cross)])
        self.deep = nn.Sequential(nn.Linear(d0, d), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Linear(d0 + d, 1)

    def forward(self, xn, xc):
        e = torch.cat([m(xc[:, i]) for i, m in enumerate(self.emb)], 1)
        x0 = torch.cat([xn, e], 1)
        x = x0
        for U, V, b in zip(self.U, self.V, self.b):
            x = x0 * ((x @ U) @ V + b) + x
        return self.head(torch.cat([x, self.deep(x0)], 1)).squeeze(1)


class SENet(nn.Module):
    """FiBiNET 의 SENET — 필드별 중요도를 배우고 재가중 후 아다마르 교차."""
    def __init__(self, n_num, cards, d=128, red=3):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, EMB) for c in cards])
        for e in self.emb:
            nn.init.normal_(e.weight, 0.0, 0.02)
        self.nf = len(cards)
        self.num_proj = nn.Linear(n_num, EMB)          # 수치부를 한 필드로
        f = self.nf + 1
        self.se = nn.Sequential(nn.Linear(f, max(f // red, 2)), nn.ReLU(),
                                nn.Linear(max(f // red, 2), f), nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(2 * f * EMB + n_num, d), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(d, d), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, xn, xc):
        e = torch.stack([m(xc[:, i]) for i, m in enumerate(self.emb)], 1)
        e = torch.cat([e, self.num_proj(xn).unsqueeze(1)], 1)      # (B,f,EMB)
        w = self.se(e.mean(2))                                     # (B,f)
        v = e * w.unsqueeze(2)
        h = torch.cat([e.flatten(1), v.flatten(1), xn], 1)
        return self.mlp(h).squeeze(1)


class DINBST(nn.Module):
    """DIN(어텐션 풀링) / BST(트랜스포머). 시퀀스는 (투수,시즌) 조회표다."""
    def __init__(self, n_num, cards, n_state, n_cell, d=128, mode="din"):
        super().__init__()
        self.mode = mode
        self.emb = nn.ModuleList([nn.Embedding(c, EMB) for c in cards])
        for e in self.emb:
            nn.init.normal_(e.weight, 0.0, 0.02)
        self.se_state = nn.Embedding(n_state, EMB)     # 과거 투구의 결과 상태
        self.se_cell = nn.Embedding(n_cell, EMB)       # 과거 투구의 상황(카운트x손)
        self.q_cell = nn.Embedding(n_cell, EMB)        # 현재 행의 상황 = 질의
        for e in (self.se_state, self.se_cell, self.q_cell):
            nn.init.normal_(e.weight, 0.0, 0.02)
        ds = 2 * EMB
        if mode == "din":
            self.att = nn.Sequential(nn.Linear(ds + EMB + ds, 64), nn.ReLU(),
                                     nn.Linear(64, 1))
        else:
            self.proj = nn.Linear(ds, ds)
            self.tr = nn.TransformerEncoderLayer(ds, 2, 64, 0.1, batch_first=True)
        d0 = n_num + len(cards) * EMB + ds
        self.mlp = nn.Sequential(nn.Linear(d0, d), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(d, 1))

    def forward(self, xn, xc, sq_state, sq_cell, sq_mask, cur_cell):
        e = torch.cat([m(xc[:, i]) for i, m in enumerate(self.emb)], 1)
        s = torch.cat([self.se_state(sq_state), self.se_cell(sq_cell)], 2)
        if self.mode == "din":
            q = self.q_cell(cur_cell).unsqueeze(1).expand(-1, s.size(1), -1)
            a = self.att(torch.cat([s, q, s * q.repeat(1, 1, 2)[:, :, :s.size(2)]], 2))
            a = a.masked_fill(~sq_mask.unsqueeze(2), -1e9).softmax(1)
            pooled = (a * s).sum(1)
        else:
            h = self.tr(self.proj(s), src_key_padding_mask=~sq_mask)
            m2 = sq_mask.unsqueeze(2).float()
            pooled = (h * m2).sum(1) / m2.sum(1).clamp(min=1.0)
        return self.mlp(torch.cat([xn, e, pooled], 1)).squeeze(1)


# ------------------------------------------------------------------ 자료
def _recover5(o):
    """복원 라벨 5범주. 0=패딩, 1=성공, 2=reverse, 3=middle, 4=그 외 실패."""
    n = o["asof_pitcher_n"].to_numpy(np.float64)
    pid = o["pitcher_id"].to_numpy()
    ok = np.r_[False, (pid[1:] == pid[:-1]) & (np.diff(n) == 1)]
    src = np.where(ok)[0]
    dst = src - 1
    lab = {}
    for nm, c in (("rev", "asof_pitcher_reverse_rate"),
                  ("mid", "asof_pitcher_middle_rate")):
        cum = o[c].to_numpy(np.float64) * n
        inc = cum[src] - cum[dst]
        v = np.round(inc)
        good = (np.abs(inc - v) < 0.25) & ((v == 0) | (v == 1))
        a = np.zeros(len(o))
        a[dst[good]] = v[good]
        lab[nm] = a
    y = o["control_success"].to_numpy(np.int64)
    return np.where(y == 1, 1, np.where(lab["rev"] > 0, 2,
                    np.where(lab["mid"] > 0, 3, 4))).astype(np.int64)


def build_inputs():
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    rid = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id"])["row_id"].to_numpy()
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    pmap = pos.reindex(rid).to_numpy()
    d = F.build(DATA, VS=VS)
    season, isf, y = d["season"], d["is_f"], d["y"].astype(np.float32)
    F44 = list(d["F44"])
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    X = Xs.to_numpy(np.float32)[pmap]
    X[:, cols.index("plat_dev")] = d["X44"][:, F44.index("plat_dev")].astype(np.float32)
    old = season <= 2022
    c4 = np.where(old & isf, 0., np.where(old & ~isf, 1.,
                                          np.where(isf, 2., 3.))).astype(np.float32)
    X = np.c_[X, c4]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES] + [X.shape[1] - 1]
    m_tr = season < VS

    ni = [j for j in range(X.shape[1]) if j not in set(ci)]
    Xc = np.zeros((len(X), len(ci)), np.int64)
    cards, uvals = [], []
    for a, j in enumerate(ci):
        u = np.unique(X[m_tr, j])
        u = u[np.isfinite(u)]
        uvals.append(u)
        p = np.searchsorted(u, X[:, j])
        p = np.clip(p, 0, max(len(u) - 1, 0))
        Xc[:, a] = np.where((p < len(u)) & (u[p] == X[:, j]), p + 1, 0)
        cards.append(len(u) + 1)
    Xn = X[:, ni].astype(np.float64)
    miss = np.isnan(Xn)
    med = np.nanmedian(Xn[m_tr], 0)
    Xn = np.where(miss, med, Xn)
    mu, sd = Xn[m_tr].mean(0), Xn[m_tr].std(0) + 1e-6
    Xn = ((Xn - mu) / sd).astype(np.float32)
    hn = miss[m_tr].any(0)
    if hn.any():
        Xn = np.concatenate([Xn, miss[:, hn].astype(np.float32)], 1)

    # ---- DIN/BST 시퀀스: (투수, 시즌) -> 직전 시즌들의 마지막 K개 투구
    src = tr.iloc[pos.reindex(rid).to_numpy().argsort()] if False else tr
    o = tr.reset_index(drop=True)
    ax = globals().get("_QAX", "cnt_hand")
    if ax == "cnt_hand":
        cell = (np.clip(o.balls_before.astype(int) * 3
                        + o.strikes_before.astype(int), 0, 11) * 2
                + o.batter_hand.astype(int))
    elif ax == "base":
        cell = (pd.factorize(o.base_state.astype(str))[0] * 3
                + o.outs_before.astype(int))
    elif ax == "inn":
        cell = (np.clip(o.inning.fillna(1).astype(int), 1, 9) * 2
                + o.batter_hand.astype(int))
    else:
        raise ValueError(ax)
    cell = np.clip(np.asarray(cell), 0, 63).astype(np.int64)
    if globals().get("_MEM5", False):
        state = _recover5(o)
    else:
        state = o.control_success.astype(int).to_numpy() + 1
    pid_o, ssn_o = o.pitcher_id.to_numpy(), o.season.to_numpy()
    pids = np.unique(pid_o)
    pidx = {p: i for i, p in enumerate(pids)}
    seasons = np.unique(ssn_o)
    sidx = {s: i for i, s in enumerate(seasons)}
    SQS = np.zeros((len(pids), len(seasons), KSEQ), np.int64)
    SQC = np.zeros((len(pids), len(seasons), KSEQ), np.int64)
    SQM = np.zeros((len(pids), len(seasons), KSEQ), bool)
    order = np.lexsort((np.arange(len(o)), ssn_o, pid_o))
    for p in pids:
        sel = order[pid_o[order] == p]
        for s in seasons:
            prev = sel[ssn_o[sel] < s][-KSEQ:]
            if len(prev) == 0:
                continue
            i, j = pidx[p], sidx[s]
            SQS[i, j, :len(prev)] = state[prev]
            SQC[i, j, :len(prev)] = cell[prev] + 1
            SQM[i, j, :len(prev)] = True
    rows_p = np.array([pidx[p] for p in pid_o])[pmap.argsort().argsort()] \
        if False else np.array([pidx[p] for p in o.pitcher_id.to_numpy()])
    rows_s = np.array([sidx[s] for s in o.season.to_numpy()])
    inv = np.empty(len(tr), np.int64)
    inv[pmap] = np.arange(len(tr))          # features44 순 -> tr 순
    SEQ_S = SQS[rows_p[inv], rows_s[inv]]
    SEQ_C = SQC[rows_p[inv], rows_s[inv]]
    SEQ_M = SQM[rows_p[inv], rows_s[inv]]
    # 이력이 아예 없는 행(2019, 신인)은 첫 칸을 연다. 전부 마스킹하면
    # 트랜스포머 어텐션이 NaN 을 낸다. 열린 칸은 패딩 임베딩이라
    # '이력 없음' 토큰으로 학습된다.
    SEQ_M[~SEQ_M.any(1), 0] = True
    CUR = (cell + 1)[inv]
    # 배치(2025)용 슬라이스 — 각 투수의 전 구간 마지막 K구
    DS, DC, DM = (np.zeros((len(pids), KSEQ), np.int64),
                  np.zeros((len(pids), KSEQ), np.int64),
                  np.zeros((len(pids), KSEQ), bool))
    for pp in pids:
        prev = order[pid_o[order] == pp][-KSEQ:]
        i = pidx[pp]
        DS[i, :len(prev)] = state[prev]
        DC[i, :len(prev)] = cell[prev] + 1
        DM[i, :len(prev)] = True
    DM[~DM.any(1), 0] = True
    print(f"  자료 X {X.shape}  수치 {Xn.shape[1]}  범주 {len(cards)}  "
          f"시퀀스 유효행 {SEQ_M.any(1).mean()*100:.1f}%  "
          f"평균길이 {SEQ_M.sum(1).mean():.1f}")
    return dict(Xn=Xn, Xc=Xc, cards=cards, y=y, season=season, isf=isf,
                m_tr=m_tr, SEQ_S=SEQ_S, SEQ_C=SEQ_C, SEQ_M=SEQ_M, CUR=CUR,
                n_state=int(max(SQS.max(), DS.max())) + 1,
                n_cell=int(CUR.max()) + 1,
                # --- 추론이 그대로 재현해야 하는 것들 ---
                P_cols=np.asarray(cols + ["abs_regime"], dtype="U64"),
                P_ci=np.asarray(ci, np.int64), P_ni=np.asarray(ni, np.int64),
                P_uvals=uvals, P_med=med, P_mu=mu, P_sd=sd, P_hn=hn,
                DEP_S=DS, DEP_C=DC, DEP_M=DM,
                PIDS=pids.astype(np.int64))


# ------------------------------------------------------------------ 학습
def fit(name, D, idx, w, seed, gate, return_model=False):
    """시드는 모델 생성 **전에** 건다."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    nn_num, cards = D["Xn"].shape[1], D["cards"]
    if name == "DCNv2":
        m = DCNv2(nn_num, cards)
    elif name == "SENet":
        m = SENet(nn_num, cards)
    else:
        m = DINBST(nn_num, cards, D["n_state"], D["n_cell"],
                   mode=("din" if name == "DIN" else "bst"))
    m = m.to(DEV)
    seq = name in ("DIN", "BST")
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    XN = torch.from_numpy(D["Xn"]).to(DEV)
    XC = torch.from_numpy(D["Xc"]).to(DEV)
    Y = torch.from_numpy(D["y"]).to(DEV)
    W = torch.from_numpy(w.astype(np.float32)).to(DEV)
    if seq:
        SS = torch.from_numpy(D["SEQ_S"]).to(DEV)
        SCq = torch.from_numpy(D["SEQ_C"]).to(DEV)
        SM = torch.from_numpy(D["SEQ_M"]).to(DEV)
        CU = torch.from_numpy(D["CUR"]).to(DEV)
    for ep in range(EPOCHS):
        m.train()
        perm = np.random.permutation(idx)
        for a in range(0, len(perm), BS):
            b = torch.from_numpy(perm[a:a + BS]).to(DEV)
            args = (XN[b], XC[b]) + ((SS[b], SCq[b], SM[b], CU[b]) if seq else ())
            z = m(*args)
            p = torch.sigmoid(z)
            per = 0.5 * nn.functional.binary_cross_entropy_with_logits(
                z, Y[b], reduction="none") + 0.5 * (p - Y[b]) ** 2
            loss = (per * W[b]).sum() / W[b].sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sch.step()
    m.eval()
    if return_model:
        return m
    out = []
    with torch.no_grad():
        for a in range(0, len(gate), 8192):
            b = torch.from_numpy(gate[a:a + 8192]).to(DEV)
            args = (XN[b], XC[b]) + ((SS[b], SCq[b], SM[b], CU[b]) if seq else ())
            out.append(torch.sigmoid(m(*args)).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    D = build_inputs()
    season, isf, y = D["season"], D["isf"], D["y"].astype(np.float64)
    gate = np.where(season == VS)[0]
    yv, isf_g = y[gate], isf[gate]
    tr_idx = np.where(D["m_tr"])[0]
    old = season <= 2022
    w = np.where(isf & old, 0.1, 1.0)
    allm = np.ones(len(yv), bool)
    sc = lambda p, m=None: F.best_shift(p[allm if m is None else m],
                                        yv[allm if m is None else m])[0]
    full_base = (VS == 2024 and
                 os.path.exists(f"{DL}/c4g_2024_c4_all_s42.npy") and
                 os.path.exists(f"{DL}/c4g_2024_c4_regular_s42.npy") and
                 os.path.exists(f"{DL}/ta2024_base.npy"))
    if full_base:
        tab = 0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy") \
            + 0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")
        cur = 0.30*np.load(f"{DL}/ta2024_base.npy").mean(0) + 0.70*tab
        base_label = "0.30 CatBoost + 0.70 TabM"
    else:
        cur = np.load(f"{DL}/h2h_{VS}_friend_s42.npy")
        base_label = "submit_41-style TabM route (CatBoost archive unavailable)"
    b0 = sc(cur)
    print(f"\n  기준 ({base_label})  {b0:.1f}   판정선 단독 880\n")
    print(f"  {'구조':8s} {'단독':>8s} {'1군':>8s} {'퓨처스':>8s} {'상관':>7s}"
          f"   {'w=0.05':>8s} {'w=0.10':>8s} {'w=0.20':>8s}  {'시간':>6s}")
    for name in MODELS:
        if name not in ("DCNv2", "SENet", "DIN", "BST"):
            raise ValueError(f"unknown CTR model: {name}")
        t0 = time.time()
        ps = []
        for br, sel in (("all", np.ones(len(tr_idx), bool)),
                        ("regular", ~isf[tr_idx]), ("futures", isf[tr_idx])):
            ps.append([fit(name, D, tr_idx[sel], w, sd, gate)
                       for sd in SEEDS])
        P = [np.where(isf_g, 0.6*ps[0][i] + 0.4*ps[2][i],
                      0.6*ps[0][i] + 0.4*ps[1][i]) for i in range(len(SEEDS))]
        p = np.mean(P, 0)
        np.save(os.path.join(DL, f"cz{VS}_{name}.npy"), np.asarray(P))
        line = (f"  {name:8s} {sc(p):8.1f} {sc(p, ~isf_g):8.1f} "
                f"{sc(p, isf_g):8.1f} {np.corrcoef(cur, p)[0,1]:7.4f}   ")
        for wt in (0.05, 0.10, 0.20):
            line += f"{sc((1-wt)*cur + wt*p) - b0:+8.1f}"
        print(line + f"  {time.time()-t0:5.0f}s", flush=True)
    print("\n  단독 880 아래는 실측으로 전부 0 아니면 음수였다 "
          "(MNCA 897.5 -> +5.1 이 유일한 양수).")
