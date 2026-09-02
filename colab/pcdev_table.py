# -*- coding: utf-8 -*-
"""배치용 조회표를 만들고 행 단위 결과를 **정확히** 재현하는지 검증한다.

왜 표여야 하나
    관문을 통과하면 제출 패키지가 test.csv 24.6만 행에 대해 편차를 계산해야
    한다. 행 단위 csv(26MB)를 넣으면 test 행에는 못 쓴다 — train row_id 기준이라.
    필요한 건 (투수, 카운트) -> 편차 조회표다. 5,318 x 5 + 780 x 1 밖에 안 된다.

규칙 4
    표는 **학습 시점에 확정**되고 트랙맨(2019~2024)만 쓴다. 추론 때는 그 행의
    pitcher_id 와 balls/strikes 로 조회만 한다. 다른 평가 행이나 평가 데이터
    분포는 일절 안 본다. 행을 섞거나 일부만 넣어도 값이 안 변한다.

검증
    표로 train 전 행을 복원해 pcdev_2025.csv.gz 와 최대차를 잰다. 0 이어야 한다.
    0 이 아니면 배치가 관문과 다른 값을 쓰게 된다 — plat_dev 때 겪은 종류의 사고다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
MAPDIR = os.path.join(ROOT, "trackman_map")
DL = os.path.join(SC, "_dl")
A = 50.0
PCM = ["br", "induced_vert_break", "extension", "rel_speed", "fb"]
BHM = ["spin_rate"]
MET = sorted(set(PCM + BHM))

bm = pd.read_csv(os.path.join(MAPDIR, "batter_map.csv"), encoding="utf-8-sig")
pm = pd.read_csv(os.path.join(MAPDIR, "pitcher_map.csv"), encoding="utf-8-sig")
tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["season", "batter_trackman_id", "pitcher_trackman_id",
                          "pitch_type_group", "balls_before", "strikes_before",
                          "pitcher_hand", "rel_speed", "spin_rate",
                          "induced_vert_break", "horz_break", "extension"])
tm["bid"] = tm["batter_trackman_id"].map(dict(zip(bm.trackman_id, bm.batter_id)))
tm["pid"] = tm["pitcher_trackman_id"].map(dict(zip(pm.trackman_id, pm.pitcher_id)))
tm["fb"] = (tm["pitch_type_group"] == "fastball").astype(float)
tm["br"] = (tm["pitch_type_group"] == "breaking").astype(float)
tm["ph"] = tm["pitcher_hand"].astype(str).str[:1].str.upper()
tm["cnt"] = tm["balls_before"].astype(str) + "-" + tm["strikes_before"].astype(str)


def dev(sub, who, extra, mets):
    s = sub.dropna(subset=[who]).copy()
    s[who] = s[who].astype(int)
    base = s.groupby(who)[mets].mean()
    cell = s.groupby([who, extra])[mets].mean()
    n = s.groupby([who, extra]).size().to_numpy()
    b = base.reindex(cell.index.get_level_values(0)).to_numpy()
    return pd.DataFrame((cell.to_numpy() - b) * (n / (n + A))[:, None],
                        index=cell.index, columns=mets)


sub = tm[tm.season < 2025]
PC = dev(sub, "pid", "cnt", PCM)
BH = dev(sub, "bid", "ph", BHM)
print(f"  조회표  pc {PC.shape}   bh {BH.shape}")

np.savez_compressed(
    os.path.join(DL, "pcdev_table.npz"),
    pc_pid=np.array([i[0] for i in PC.index], np.int32),
    pc_cnt=np.array([i[1] for i in PC.index]),
    pc_val=PC.to_numpy(np.float32), pc_met=np.array(PCM),
    bh_bid=np.array([i[0] for i in BH.index], np.int32),
    bh_ph=np.array([i[1] for i in BH.index]),
    bh_val=BH.to_numpy(np.float32), bh_met=np.array(BHM))
sz = os.path.getsize(os.path.join(DL, "pcdev_table.npz"))
print(f"  저장  pcdev_table.npz  {sz/1024:.0f} KB")

# ------------------------------------------------------------------ 검증
z = np.load(os.path.join(DL, "pcdev_table.npz"), allow_pickle=False)
LPC = {(int(p), str(c)): i for i, (p, c) in
       enumerate(zip(z["pc_pid"], z["pc_cnt"]))}
LBH = {(int(b), str(h)): i for i, (b, h) in
       enumerate(zip(z["bh_bid"], z["bh_ph"]))}


def lookup(pid, bid, balls, strikes, phand):
    """추론 때 쓰는 함수. **그 행의 값만** 본다 (규칙 4)."""
    n = len(pid)
    out = np.zeros((n, len(PCM) + len(BHM)), np.float32)
    cnt = np.char.add(np.char.add(balls.astype(str), "-"), strikes.astype(str))
    ip = np.array([LPC.get((int(a), str(b)), -1) for a, b in zip(pid, cnt)])
    ok = ip >= 0
    out[ok, :len(PCM)] = z["pc_val"][ip[ok]]
    ib = np.array([LBH.get((int(a), str(b)), -1) for a, b in zip(bid, phand)])
    ok = ib >= 0
    out[ok, len(PCM):] = z["bh_val"][ib[ok]]
    return out


import features44 as F                                          # noqa: E402
d = F.build(DATA, VS=2025, return_frame=True)
fr = d["frame"]
raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["row_id", "balls_before", "strikes_before",
                           "pitcher_hand"]).set_index("row_id").reindex(
    fr["row_id"].to_numpy())
got = lookup(fr["pitcher_id"].to_numpy(), fr["batter_id"].to_numpy(),
             raw["balls_before"].to_numpy(), raw["strikes_before"].to_numpy(),
             raw["pitcher_hand"].map({1: "L", 2: "R"}).fillna("R").to_numpy())
ref = pd.read_csv(os.path.join(DL, "pcdev_2025.csv.gz"))
names = [f"pc_{m}" for m in PCM] + [f"bh_{m}" for m in BHM]
print(f"\n  {'열':24s} {'최대차':>12s}")
worst = 0.0
for j, nm in enumerate(names):
    dd = float(np.abs(got[:, j] - ref[nm].to_numpy(np.float32)).max())
    worst = max(worst, dd)
    print(f"  {nm:24s} {dd:12.3e}")
print(f"\n  전체 최대차 {worst:.3e}")
print("  " + ("표가 행 단위 결과를 그대로 재현한다. 배치 경로 확보."
               if worst < 1e-6 else
               "**어긋난다.** 배치가 관문과 다른 값을 쓰게 된다."))

# ---- 규칙 4 감사: 순서 섞기 / 부분 집합
rng = np.random.default_rng(0)
per = rng.permutation(len(fr))
g2 = lookup(fr["pitcher_id"].to_numpy()[per], fr["batter_id"].to_numpy()[per],
            raw["balls_before"].to_numpy()[per],
            raw["strikes_before"].to_numpy()[per],
            raw["pitcher_hand"].map({1: "L", 2: "R"}).fillna("R").to_numpy()[per])
d1 = float(np.abs(g2 - got[per]).max())
s = per[:len(per) // 5]
g3 = lookup(fr["pitcher_id"].to_numpy()[s], fr["batter_id"].to_numpy()[s],
            raw["balls_before"].to_numpy()[s], raw["strikes_before"].to_numpy()[s],
            raw["pitcher_hand"].map({1: "L", 2: "R"}).fillna("R").to_numpy()[s])
d2 = float(np.abs(g3 - got[s]).max())
print(f"\n  규칙 4 - 순서 섞기 최대차 {d1:.3e}   20% 부분집합 최대차 {d2:.3e}")
print("  " + ("통과. 행마다 독립이다." if max(d1, d2) < 1e-9 else "**위반이다.**"))
