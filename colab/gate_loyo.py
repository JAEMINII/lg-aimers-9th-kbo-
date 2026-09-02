# -*- coding: utf-8 -*-
"""게이팅 재검 (강화판) — leave-one-year-out, 3폴드, 4원 구성원.

로컬 1차 실험의 약점: 게이트를 **한 해**로 학습해서 그 해 패턴에 과적합했다
(인폴드 +11~23, 교차 -3.5~-6.4). 이번엔 **두 해로 학습 -> 남은 한 해 채점**을
세 방향 전부 돈다. 두 해를 보면 연도 불변 패턴만 남을 수 있다 — 게이팅이
진짜 죽었는지 가리는 더 공정한 시험. (재민님: "여기다가 다시 해봐")

사전 단계 (이 스크립트가 직접 만든다, 없으면):
    dg_2023_DIN.npy   ctr_zoo DIN, VS=2023, 3브랜치 x 3시드 라우팅
    cb50_2023.npy     50열 CatBoost 라우팅, VS=2023, 3시드 평균 (GPU)

게이트  표준화 44열 -> MLP(32) -> softmax 4가중. 자유형 + 보수형(고정가중 주변).
        행 내부 함수만 사용 — 규칙 4 적합.
기준    고정 (0.31/0.13/0.31/0.25). 최적 시프트로 채점.
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "nn_experiments"))
os.environ.setdefault("AIMERS_ROOT", ROOT)
DATA = os.environ.get("AIMERS_DATA", os.path.join(ROOT, "open (1)", "data"))
DL = os.path.join(SC, "_dl")
FIX = np.array([0.31, 0.13, 0.31, 0.25])
FOLDS = (2022, 2023, 2024)

import features44 as F                                          # noqa: E402


def ensure_din_2023():
    f = os.path.join(DL, "dg_2023_DIN.npy")
    if os.path.exists(f):
        return
    import ctr_zoo as Z
    Z.VS = 2023
    D = Z.build_inputs()
    season, isf = D["season"], D["isf"]
    gate = np.where(season == 2023)[0]
    tr = np.where(D["m_tr"])[0]
    w = np.where(isf & (season <= 2022), 0.1, 1.0)
    P = []
    for sd in (42, 1, 777):
        ps = [Z.fit("DIN", D, tr[sel], w, sd, gate)
              for sel in (np.ones(len(tr), bool), ~isf[tr], isf[tr])]
        isf_g = isf[gate]
        P.append(np.where(isf_g, 0.6 * ps[0] + 0.4 * ps[2],
                          0.6 * ps[0] + 0.4 * ps[1]))
        print(f"  DIN2023 seed {sd} 완료", flush=True)
    np.save(f, np.asarray(P))


def ensure_cb50_2023():
    f = os.path.join(DL, "cb50_2023.npy")
    if os.path.exists(f):
        return
    import train_c12_submit as T
    from catboost import CatBoostClassifier
    built = F.build(DATA, VS=2023, return_frame=True)
    fr = built["frame"]
    base_f = list(built["F44"])
    fr, _ = T.add_c12(fr, return_tables=True)
    fr, _ = T.add_cmh(fr, return_tables=True)
    feats = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                      "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
    X = fr[feats].to_numpy(np.float32)
    y = fr["control_success"].to_numpy(np.float64)
    season = fr["season"].to_numpy()
    gt = fr["game_type"].to_numpy()
    tr_m, te_m = season < 2023, season == 2023
    w = 2.0 ** (season[tr_m].astype(np.float64) - 2019.0)
    isf_t = gt[te_m] == 1
    hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
              verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
    masks = {"all": np.ones(int(tr_m.sum()), bool),
             "regular": gt[tr_m] == 0, "futures": gt[tr_m] == 1}
    pg = {g: [] for g in masks}
    for g, m in masks.items():
        for sd in (1, 42, 777):
            mo = CatBoostClassifier(random_seed=sd, **hp)
            mo.fit(X[tr_m][m], y[tr_m][m], sample_weight=w[m])
            pg[g].append(mo.predict_proba(X[te_m])[:, 1])
        print(f"  CB2023 {g} 완료", flush=True)
    ps = [np.where(isf_t, 0.6 * pg["all"][i] + 0.4 * pg["futures"][i],
                   0.6 * pg["all"][i] + 0.4 * pg["regular"][i])
          for i in range(3)]
    np.save(f, np.mean(ps, 0).astype(np.float32))


def fold_data(vs, Xstat):
    d = F.build(DATA, VS=vs)
    g = np.where(d["season"] == vs)[0]
    y = d["y"].astype(np.float64)[g]
    X = d["X44"].astype(np.float64)[g]
    X = np.where(np.isnan(X), Xstat["med"], X)
    X = ((X - Xstat["mu"]) / Xstat["sd"]).astype(np.float32)
    if vs == 2024:
        cb = np.load(f"{DL}/ta2024_base.npy").mean(0)
        tab = 0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy") \
            + 0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")
    elif vs == 2022:
        cb = np.load(f"{DL}/cb50fixed_2022.npy").astype(np.float64)
        tab = np.load(f"{DL}/h2h_2022_friend_s42.npy").astype(np.float64)
    else:
        cb = np.load(f"{DL}/cb50_2023.npy").astype(np.float64)
        tab = np.load(f"{DL}/h2h_2023_friend_s42.npy").astype(np.float64)
    din = np.load(f"{DL}/dg_{vs}_DIN.npy").mean(0)
    ms = np.load(f"{DL}/msaudit/audit_all_vs{vs}_direct.npy").astype(np.float64)
    P = np.column_stack([cb, tab, din, ms]).astype(np.float32)
    return X, P, y


class Gate(nn.Module):
    def __init__(self, n_in, conservative):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(32, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.cons = conservative
        self.register_buffer("logfix",
                             torch.log(torch.tensor(FIX, dtype=torch.float32)))

    def forward(self, x):
        z = self.net(x)
        if self.cons:
            z = z + self.logfix
        return torch.softmax(z, 1)


def train_gate(X, P, y, conservative, seed=42, epochs=5, bs=8192, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Gate(X.shape[1], conservative).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    XT = torch.from_numpy(X).to(dev)
    PT = torch.from_numpy(P).to(dev)
    YT = torch.from_numpy(y.astype(np.float32)).to(dev)
    n = len(y)
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for a in range(0, n, bs):
            b = torch.from_numpy(perm[a:a + bs]).to(dev)
            g = m(XT[b])
            p = (g * PT[b]).sum(1)
            loss = ((p - YT[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m, dev


@torch.no_grad()
def apply_gate(m, dev, X, P):
    out = []
    for a in range(0, len(X), 16384):
        g = m(torch.from_numpy(X[a:a + 16384]).to(dev))
        out.append((g * torch.from_numpy(P[a:a + 16384]).to(dev)).sum(1)
                   .cpu().numpy())
    return np.concatenate(out).astype(np.float64)


if __name__ == "__main__":
    ensure_din_2023()
    ensure_cb50_2023()
    # 표준화 통계는 전 학습기간(2019~2024)에서 — 행 내부 결정론
    d = F.build(DATA, VS=2025)
    Xa = d["X44"].astype(np.float64)
    Xstat = dict(med=np.nanmedian(Xa, 0), mu=None, sd=None)
    Xf = np.where(np.isnan(Xa), Xstat["med"], Xa)
    Xstat["mu"], Xstat["sd"] = Xf.mean(0), Xf.std(0) + 1e-9
    D = {vs: fold_data(vs, Xstat) for vs in FOLDS}
    sc = lambda p, y: F.best_shift(p, y)[0]
    print("\n  단독/고정 기준 확인")
    for vs in FOLDS:
        X, P, y = D[vs]
        print(f"    VS={vs}  cb {sc(P[:,0],y):8.1f}  tab {sc(P[:,1],y):8.1f}  "
              f"din {sc(P[:,2],y):8.1f}  ms {sc(P[:,3],y):8.1f}   "
              f"고정혼합 {sc(P @ FIX, y):8.1f}")
    print("\n" + "=" * 88)
    print("  leave-one-year-out 게이팅 — 두 해 학습, 남은 해 채점")
    print("=" * 88)
    for hold in FOLDS:
        tr_vs = [v for v in FOLDS if v != hold]
        Xs = np.concatenate([D[v][0] for v in tr_vs])
        Ps = np.concatenate([D[v][1] for v in tr_vs])
        ys = np.concatenate([D[v][2] for v in tr_vs])
        Xt, Pt, yt = D[hold]
        ref = sc(Pt @ FIX, yt)
        print(f"  학습 {tr_vs} -> 채점 {hold}   고정 {ref:.1f}")
        for nm, cons in (("자유", False), ("보수", True)):
            outs = []
            for sd in (42, 1, 777):
                m, dev = train_gate(Xs, Ps, ys, cons, seed=sd)
                outs.append(sc(apply_gate(m, dev, Xt, Pt), yt))
            mu = float(np.mean(outs))
            print(f"    {nm} 게이트  {mu:8.1f}  (시드별 "
                  + " ".join(f"{v:.1f}" for v in outs)
                  + f")   고정 대비 {mu-ref:+.1f}", flush=True)
