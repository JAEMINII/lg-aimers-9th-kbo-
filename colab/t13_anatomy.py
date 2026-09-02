# -*- coding: utf-8 -*-
"""team13 해부 — 투수측/타자측/홈원정/월별 지속성. 로컬 train.csv 만."""
import sys, numpy as np, pandas as pd
W = sys.stdout.buffer.write
def P(s): W((s + "\n").encode("utf-8"))
tr = pd.read_csv("open (1)/data/train.csv", encoding="utf-8-sig",
                 usecols=["season","game_month","game_type","top_bottom",
                          "pitcher_team_id","batter_team_id","pitcher_id",
                          "control_success"])
gt = tr["game_type"].astype(str)
tr["isf"] = (gt=="F")|(gt=="1")
y = tr["control_success"].astype(float)
p13 = tr["pitcher_team_id"]==13; b13 = tr["batter_team_id"]==13
tb = tr["top_bottom"].astype(str).str.upper().str[0]
home13 = np.where(tb=="T", tr["pitcher_team_id"], tr["batter_team_id"])==13
inv = p13|b13
post = (tr["season"]>2023)|((tr["season"]==2023)&((tr["game_month"]>=5)|tr["isf"]))
half = np.where(tr["game_month"]<=6,"H1","H2")

P("== 1) 시즌x반기, R행만: t13관여 vs 비관여 성공률 (n)")
m = ~tr["isf"]
g = tr[m].assign(y=y[m], inv=inv[m], half=half[m]).groupby(["season","half","inv"])["y"].agg(["mean","size"]).unstack("inv")
for (s,h),row in g.iterrows():
    if s<2021: continue
    d = row[("mean",True)]-row[("mean",False)]
    P(f"  {s}{h}  비관여 {row[('mean',False)]:.4f}  관여 {row[('mean',True)]:.4f}  d={d:+.4f}  n13={int(row[('size',True)])}")
P("== 2) 같은 분해, F행만")
m = tr["isf"].to_numpy()
g = tr[m].assign(y=y[m], inv=inv[m], half=half[m]).groupby(["season","half","inv"])["y"].agg(["mean","size"]).unstack("inv")
for (s,h),row in g.iterrows():
    if s<2021: continue
    try: d = row[("mean",True)]-row[("mean",False)]
    except: continue
    if np.isnan(d): continue
    P(f"  {s}{h}  비관여 {row[('mean',False)]:.4f}  관여 {row[('mean',True)]:.4f}  d={d:+.4f}  n13={int(row[('size',True)])}")
P("== 3) R행 2023-2024 월별: 13투수측 / 13타자측 / 13홈 / 리그(비관여)")
m = (~tr["isf"]) & (tr["season"]>=2023)
sub = tr[m].assign(y=y[m])
for s in (2023,2024):
    for mo in sorted(sub[sub.season==s]["game_month"].unique()):
        mm = (sub.season==s)&(sub.game_month==mo)
        rows = sub[mm]
        lg = rows[~inv[m][mm.to_numpy()[:0].size and mm]|True]  # placeholder
        i = mm.to_numpy()
        base = rows[~(p13|b13)[m][i[i]] if False else ~((rows.pitcher_team_id==13)|(rows.batter_team_id==13))]["y"].mean()
        a = rows[rows.pitcher_team_id==13]["y"].mean()
        b = rows[rows.batter_team_id==13]["y"].mean()
        h13 = np.where(rows.top_bottom.astype(str).str.upper().str[0]=="T", rows.pitcher_team_id, rows.batter_team_id)==13
        hm = rows[h13]["y"].mean(); aw = rows[((rows.pitcher_team_id==13)|(rows.batter_team_id==13))&~h13]["y"].mean()
        P(f"  {s}-{int(mo):02d}  리그 {base:.3f}  13투 {a:.3f}({a-base:+.3f})  13타 {b:.3f}({b-base:+.3f})  13홈 {hm:.3f}({hm-base:+.3f})  13원정 {aw:.3f}({aw-base:+.3f})")
