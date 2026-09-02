# -*- coding: utf-8 -*-
"""시즌가중과 볼카운트 가중을 **as-of 피처 위에서, 두 폴드로** 다시 잰다.

왜 다시 재나
    시즌가중을 전에 기각했다 (decay 1.2/1.5/2.0 전부 단조 악화, 0/3).
    그런데 그 측정에는 결함이 둘 있었다.

        ① 폴드가 하나였다 (VS=2024). '한 폴드 관문은 그 해에 과적합한다.'
        ② **누출된 plat_dev 위에서 쟀다.** 그 열은 표적상관이 0.024 로
           부풀려져 있었고 (as-of 는 0.015), 모델이 과신하도록 학습돼 있었다.

    ②가 특히 걸린다. 시즌가중은 '무엇을 얼마나 믿을 것인가' 를 바꾸는 축인데,
    믿음의 대상 하나가 망가져 있었으면 결론이 달라질 수 있다. 스케줄러를 고친
    뒤 임베딩 기각이 무효가 됐던 것과 같은 상황이다.

    지금 배치(submit_30, LB 1069)는 as-of 피처를 쓴다. 그 위에서 다시 재야 한다.

볼카운트 가중 — 한 번도 안 재봄
    0-0 이 전체의 25.8%, 3-0 이 1.3% 다. 균등 가중이면 0-0 이 지배한다.
    빈도의 역수(제곱근)로 가중하면 희귀 카운트에 무게가 실린다.

    기전을 정직하게 적어두면 — 카운트 분포는 해마다 안정적이라 시즌가중이
    통했던 '공변량 이동' 논리가 여기엔 없다. 다만 손실의 무게 배분이 바뀌면
    모델이 어디에 용량을 쓸지가 달라진다. 그건 안 재봤다.

    같은 부류였던 relw(신뢰도 가중)가 -7.3, 0/3 으로 졌다. 사전확률은 낮다.

팔 (전부 as-of plat_dev, 옛퓨처스 가중 0.1 은 공통)
    base       추가 가중 없음                          현행
    d1.5       시즌 decay 1.5
    d2.0       시즌 decay 2.0
    cnt_inv    볼카운트 빈도의 역제곱근으로 가중

폴드
    VS=2022, VS=2024.  2023 은 퓨처스 체제전환년이라 폴드가 깨진다 (전체 점수 5.9).
    두 폴드에서 같은 부호로 이겨야 믿는다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
FOLDS = (2022, 2024)
ARMS = ("base", "d1.5", "d2.0", "cnt_inv")

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def one(seed, tr_idx, t_isf, w, season, VS, gate, isf_g):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"VS{VS} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        pr = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=pr, seed=seed + e,
                    tag=f"VS{VS} {br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(isf_g, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


if __name__ == "__main__":
    d0 = F.build(DATA, VS=2024)
    season = d0["season"].astype(np.float64)
    isf, y = d0["is_f"], d0["y"].astype(np.float64)
    F44 = list(d0["F44"])
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id", "balls_before", "strikes_before"])
    cm = (raw["balls_before"].to_numpy() * 3
          + raw["strikes_before"].to_numpy()).astype(np.int64)

    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(raw["row_id"].to_numpy()).to_numpy()

    ALL = {}
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        G.gate, G.yv = gate, yv
        m_tr = season < VS

        hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
        Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
        cols = list(Xs.columns)
        Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
        ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        # plat_dev 를 as-of 로 (submit_30 과 같게)
        dv = F.build(DATA, VS=VS)
        Xfr[:, cols.index("plat_dev")] = \
            dv["X44"][:, F44.index("plat_dev")].astype(np.float32)

        old = season <= OLD_F_MAX
        c4 = np.where(old & isf, 0.0,
                      np.where(old & ~isf, 1.0,
                               np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
        Xn, Xc, cards = G.prep(np.concatenate([Xfr, c4], 1), m_tr,
                               ci + [Xfr.shape[1]])
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        base_w = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        s_tr = season[tr_idx]
        cnt = cm[tr_idx]
        freq = np.bincount(cnt, minlength=cnt.max() + 1).astype(np.float64)
        inv = 1.0 / np.sqrt(np.maximum(freq[cnt], 1.0))
        inv = inv / inv.mean()          # 평균 1 로 정규화

        W = {"base": base_w,
             "d1.5": base_w * (1.5 ** (s_tr - 2019)),
             "d2.0": base_w * (2.0 ** (s_tr - 2019)),
             "cnt_inv": base_w * inv}
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}")
        G.log(f"    카운트 가중 범위 {inv.min():.3f} ~ {inv.max():.3f}  "
              f"(0-0 {inv[cnt == 0][0]:.3f}, 3-0 {inv[cnt == 9][0]:.3f})")

        for nm in ARMS:
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, W[nm], season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"ow_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:8s} 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 90)
    print("  as-of 피처 위에서 다시 잰 시즌가중 / 볼카운트가중")
    print("=" * 90)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:8s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)

    print("\n  이전 측정(누출 피처, 폴드 1개)에서는 시즌가중이 -10.8 ~ -23.9, 0/3 이었다.")
    print("  as-of 위에서 부호가 바뀌면 그 기각은 누출 탓이었던 것이다.")
    print("  두 폴드에서 같은 부호로 +5 이상이어야 옮긴다.")
