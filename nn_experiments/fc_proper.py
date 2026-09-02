# -*- coding: utf-8 -*-
"""
FC 신경망 재시험 — 이번엔 제대로.

왜 다시 하는가
  과거 FC 는 2024 관문 −251 로 크게 졌고, 그걸 근거로 "신경망은 이 문제에 안 맞는다"
  고 두 번 결론냈다. 그런데 CatBoost 도 −539 로 기각했다가 설정 실수였음이 드러났다.
  신경망은 GBDT 보다 설정에 훨씬 민감하다. 같은 함정일 수 있다.

  그리고 리더보드가 방금 알려준 것: 147만 행에서는 sklearn 과 CatBoost 가 동점(989/987)이다.
  GBDT 안에서 알고리즘을 바꿔봐야 얻을 게 없다. 근사 방식이 근본적으로 다른
  모델이라야 앙상블 이득이 나온다. 폴드2023(퓨처스 붕괴)에서 FC 가 GBDT 를
  크게 이겼던 것이 그 근거다.

과거 FC 에서 확인되지 않았던 것 — 이번에 전부 명시한다
  1) 연속형 표준화     피처 스케일이 제각각이다 (li 0~5, 승률 0~100, 비율 0~1)
  2) 범주형 임베딩     pitcher_id 792개를 수치로 넣으면 '번호 순서'에 의미를 부여하게 된다
  3) 결측 처리         asof_* 는 신인에게 NaN. 중앙값 대치 + 결측 표시자
  4) 미지 범주         2024 에 처음 나온 투수는 학습에 없던 값. 0번(unknown)으로
  5) 조기종료          학습 구간의 마지막 5% 를 내부 검증으로. 관문 연도는 절대 안 본다
  6) 학습률 스케줄      OneCycle. 고정 lr 은 약신호 문제에서 수렴이 나쁘다

측정
  관문 2019~2023 -> 2024, 1군 채점(2025 는 전부 1군), shift 0 / −0.05 둘 다.
  같은 관문 GBDT 실측 (1군, shift 0 / −0.05):
      sklearn  전체단독  814.2 / 856.2
      CatBoost 전체단독  844.5 / 876.5
      CatBoost 60+40     850.6 / 886.3
  FC 가 이 근처면 앙상블 재료로 값어치가 있다. −100 이상 벌어지면 축을 닫는다.
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(4)
ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
SC = os.path.dirname(os.path.abspath(__file__))
TGT = "control_success"
SEEDS = (1, 42, 777)
ALPHA, ALPHA_PLAT = 50.0, 300.0
VS = 2024
def H(t):
    print("\n" + "=" * 90); print(t); print("=" * 90); sys.stdout.flush()

t00 = time.time()
tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
tr["_r"] = tr.row_id.str.slice(6).astype("int32")
tr = tr.sort_values("_r").reset_index(drop=True)
IS_F = (tr.game_type.values == "F")
for c in tr.columns:
    if tr[c].dtype == "float64":
        tr[c] = tr[c].astype("float32")
PN, BN, MX = "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"
tr["c_p"] = (tr.asof_pitcher_success_rate.fillna(0) * tr[PN]).round()
tr["c_b"] = (tr.asof_batter_success_rate.fillna(0) * tr[BN]).round()
f1 = tr.groupby(["pitcher_id", "season"], sort=False).head(1)[["pitcher_id", "season", PN, "c_p"]]
f1.columns = ["pitcher_id", "season", "bn_", "bs_"]
tr = tr.merge(f1, on=["pitcher_id", "season"], how="left")
f2 = tr.groupby(["batter_id", "season"], sort=False).head(1)[["batter_id", "season", BN, "c_b"]]
f2.columns = ["batter_id", "season", "bbn_", "bbs_"]
tr = tr.merge(f2, on=["batter_id", "season"], how="left")
tr["pn_cur"] = (tr[PN] - tr.bn_).clip(lower=0).astype("float32")
tr["p_is_succ"] = (((tr.c_p - tr.bs_).clip(lower=0) + ALPHA * .5)
                   / (tr.pn_cur + ALPHA)).astype("float32")
tr["b_is_succ"] = (((tr.c_b - tr.bbs_).clip(lower=0) + ALPHA * .5)
                   / ((tr[BN] - tr.bbn_).clip(lower=0) + ALPHA)).astype("float32")
tr.drop(columns=["c_p", "c_b", "bn_", "bs_", "bbn_", "bbs_"], inplace=True)
gc.collect()

PC = tr.groupby(["pitcher_id", "season", "batter_hand"])[TGT].agg(["size", "sum"]).unstack(fill_value=0)
PC.columns = [f"{x}{int(h)}" for x, h in PC.columns]
for h in (1, 2):
    for x in ("size", "sum"):
        if f"{x}{h}" not in PC.columns:
            PC[f"{x}{h}"] = 0
PCC = PC.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.)
PCC["n_all"] = PCC.size1 + PCC.size2
PCC["s_all"] = PCC.sum1 + PCC.sum2
PH = tr.groupby("pitcher_id").pitcher_hand.first()
BHAND = tr.batter_hand.values.copy()
for c, m in {"top_bottom": {"T": 0, "B": 1}, "game_type": {"R": 0, "F": 1},
             "base_state": {"___": 0, "1__": 1, "_2_": 2, "__3": 3,
                            "12_": 4, "1_3": 5, "_23": 6, "123": 7}}.items():
    tr[c] = tr[c].map(m).fillna(-1).astype("int8")
tr["cm"] = ((tr.balls_before * 3 + tr.strikes_before) * 4
            + tr.pitcher_hand * 2 + tr.batter_hand).astype("int16")

m_tr = (tr.season < VS).values
m_va = (tr.season == VS).values
hist = tr[m_tr]
gm_ = float(hist[TGT].mean())
lgph = hist.groupby(["pitcher_hand", "batter_hand"])[TGT].mean()
lgp = hist.groupby("pitcher_hand")[TGT].mean()
lg48 = hist.groupby("cm")[TGT].mean() - gm_
h2 = tr[m_tr & tr.p_is_succ.notna().values]
def slope(x, y):
    v = x.var()
    return np.cov(x, y)[0, 1] / v if v > 1e-12 else np.nan
rel48 = (h2.groupby("cm").apply(lambda d: slope(d.p_is_succ.values, d[TGT].values),
                                include_groups=False)
         / slope(h2.p_is_succ.values, h2[TGT].values)).clip(.2, 1.8)
del hist, h2
gc.collect()
tr["lg_cm_eff"] = tr.cm.map(lg48).fillna(0.).astype("float32")
tr["cm_rel"] = tr.cm.map(rel48).fillna(1.).astype("float32")
tr["p_adj_cm"] = (gm_ + (tr.p_is_succ - gm_) * tr.cm_rel).astype("float32")
ph_of = PH.reindex(PCC.index.get_level_values("pitcher_id")).values
Pd = {}
for h in (1, 2):
    pr = np.array([lgph.get((p_, h), gm_) for p_ in ph_of])
    Pd[h] = (PCC[f"sum{h}"].values + ALPHA_PLAT * pr) / (PCC[f"size{h}"].values + ALPHA_PLAT)
pa = ((PCC.s_all.values + ALPHA_PLAT * np.array([lgp.get(p_, gm_) for p_ in ph_of]))
      / (PCC.n_all.values + ALPHA_PLAT))
lut = pd.DataFrame({"p1": Pd[1], "p2": Pd[2], "pa": pa}, index=PCC.index).reindex(
    pd.MultiIndex.from_arrays([tr.pitcher_id.values, tr.season.values]))
dv = np.where(BHAND == 1, lut.p1.values, lut.p2.values) - lut.pa.values
tr["plat_dev"] = np.where(np.isnan(dv), 0., dv).astype("float32")

test_cols = pd.read_csv(os.path.join(D, "test.csv"), encoding="utf-8-sig", nrows=0).columns
BASE = [c for c in test_cols if c != "row_id"]
DUP = [MX, "away_win_expectancy", "run_total_before", "score_diff_home",
       "num_runners_on", "runner_on_1b", "runner_on_2b", "runner_on_3b"]
F44 = ([c for c in BASE if c not in DUP + [PN, BN]]
       + ["p_is_succ", "pn_cur", "b_is_succ", "lg_cm_eff", "cm_rel", "p_adj_cm", "plat_dev"])

# 범주는 '값의 크기에 의미가 없는' 것. pitcher_id 3번이 2번보다 크다는 건 뜻이 없다.
# 수치로 넣으면 MLP 가 그 순서에 의미를 부여하려 든다. 임베딩이 맞다.
CAT = [c for c in ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
                   "base_state", "pitcher_hand", "batter_hand", "top_bottom", "game_type",
                   "balls_before", "strikes_before", "outs_before", "inning",
                   "season", "game_month", "game_dayofweek"] if c in F44]
CON = [c for c in F44 if c not in CAT]
print(f"피처 {len(F44)}개 = 범주 {len(CAT)} + 연속 {len(CON)}")

cat_idx, cat_card = [], []
for c in CAT:
    vals = pd.unique(tr.loc[m_tr, c])
    mp = {v: i + 1 for i, v in enumerate(sorted(v for v in vals if pd.notna(v)))}
    cat_idx.append(tr[c].map(mp).fillna(0).astype("int32").values)
    cat_card.append(len(mp) + 1)
Xc = np.stack(cat_idx, 1)
print("  카디널리티:", {c: n for c, n in zip(CAT, cat_card)})

Xn = tr[CON].to_numpy(dtype=np.float32)
miss = np.isnan(Xn)
has_nan = miss[m_tr].any(0)
med = np.nanmedian(Xn[m_tr], 0)
Xn = np.where(miss, med, Xn)
mu, sd = Xn[m_tr].mean(0), Xn[m_tr].std(0) + 1e-6
Xn = ((Xn - mu) / sd).astype(np.float32)
if has_nan.any():
    Xn = np.concatenate([Xn, miss[:, has_nan].astype(np.float32)], 1)
    print(f"  결측 표시자 {int(has_nan.sum())}개 추가 -> 연속 입력 {Xn.shape[1]}차원")
y = tr[TGT].to_numpy(dtype=np.float32)
del tr
gc.collect()


class FC(nn.Module):
    def __init__(self, cards, n_con, hid=(512, 256), p=0.1):
        super().__init__()
        # 임베딩 차원: 카디널리티의 0.4 제곱 근처. 792명 -> 15, 4개 -> 2
        dims = [max(2, min(32, int(round(c ** 0.4)))) for c in cards]
        self.emb = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cards, dims)])
        d_in = sum(dims) + n_con
        layers, d = [], d_in
        for h in hid:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)
        self.d_in = d_in

    def forward(self, xc, xn):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.emb)]
        return self.mlp(torch.cat(e + [xn], 1)).squeeze(1)


def bss(p, yy):
    r = yy.mean()
    return 100000 * (1 - ((p - yy) ** 2).mean() / (r * (1 - r)))
def sh(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))

idx_tr = np.where(m_tr)[0]
# SUB_N 이 설정되면 그만큼만 균등 표본으로 학습한다.
# pytabkit 공식 구현이 20만 행에서 낸 점수(MLP-PLR 607.1 / RealMLP 516.1)와
# 같은 조건으로 비교해, 내 손구현에 실제 결함이 있는지 판단하기 위한 장치다.
_sub = int(os.environ.get("SUB_N", "0"))
if _sub and _sub < len(idx_tr):
    idx_tr = np.sort(np.random.RandomState(0).choice(idx_tr, _sub, replace=False))
    print(f"부분표본 {_sub:,} 행으로 학습 (공식구현 비교용)")
cut = int(len(idx_tr) * 0.95)
i_fit, i_val = idx_tr[:cut], idx_tr[cut:]
i_gate = np.where(m_va)[0]
fv = IS_F[m_va]
yv = y[i_gate]
H(f"관문 학습 {len(i_fit):,} (+내부검증 {len(i_val):,}) -> 2024 {len(i_gate):,}")

def to_t(idx):
    return (torch.from_numpy(Xc[idx].astype(np.int64)),
            torch.from_numpy(Xn[idx]), torch.from_numpy(y[idx]))

Tc_f, Tn_f, Ty_f = to_t(i_fit)
Tc_v, Tn_v, Ty_v = to_t(i_val)
Tc_g, Tn_g, _ = to_t(i_gate)

EPOCHS, BS = 12, 4096
preds = []
for sd_ in SEEDS:
    torch.manual_seed(sd_)
    np.random.seed(sd_)
    net = FC(cat_card, Xn.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    nstep = (len(i_fit) + BS - 1) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EPOCHS,
                                                steps_per_epoch=nstep)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = 1e9, None, 0
    t1 = time.time()
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(i_fit))
        for b in range(nstep):
            j = perm[b * BS:(b + 1) * BS]
            opt.zero_grad()
            loss = lossf(net(Tc_f[j], Tn_f[j]), Ty_f[j])
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Tc_v, Tn_v)).numpy()
        vl = float(((pv - Ty_v.numpy()) ** 2).mean())      # 내부검증 Brier
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f"    seed{sd_} ep{ep+1:02d} 내부Brier {vl:.6f}{'  *' if bad == 0 else ''}",
              flush=True)
        if bad >= 3:
            print(f"    조기종료 (ep{ep+1})")
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pg = torch.cat([torch.sigmoid(net(Tc_g[i:i + 65536], Tn_g[i:i + 65536]))
                        for i in range(0, len(i_gate), 65536)]).numpy()
    preds.append(pg)
    print(f"  seed{sd_} 완료 {time.time()-t1:.0f}s  단일 1군 {bss(pg[~fv], yv[~fv]):.1f}",
          flush=True)

H("결과 — 관문 1군 채점")
p = np.mean(preds, 0)
for c in (0.0, -0.05):
    q = sh(p, c)
    print(f"  FC {len(SEEDS)}시드 앙상블  shift {c:+.2f}   "
          f"1군 {bss(q[~fv], yv[~fv]):8.1f}   전체 {bss(q, yv):8.1f}")
print("\n  같은 관문 GBDT 실측 (1군, shift 0 / −0.05)")
print("    sklearn  전체단독   814.2 / 856.2")
print("    CatBoost 전체단독   844.5 / 876.5")
print("    CatBoost 60+40      850.6 / 886.3")
np.save(os.path.join(SC, "fc_gate_pred.npy"), p)
print("\n  예측 저장: fc_gate_pred.npy (앙상블 실험용)")
print(f"\n총 {time.time()-t00:.0f}s")
