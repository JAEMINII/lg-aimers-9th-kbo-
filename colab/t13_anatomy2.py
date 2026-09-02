# -*- coding: utf-8 -*-
import sys, numpy as np, pandas as pd
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8"))
tr = pd.read_csv("open (1)/data/train.csv", encoding="utf-8-sig",
                 usecols=["season","game_month","game_type","top_bottom",
                          "pitcher_team_id","batter_team_id","pitcher_id","control_success"])
gt = tr["game_type"].astype(str)
P("game_type 값: " + str(sorted(gt.unique())))
P("13팀 F행 수: " + str(int(((tr.pitcher_team_id==13)|(tr.batter_team_id==13)) [gt!='R'].sum() if 'R' in set(gt.unique()) else -1)))
isf = ~(gt=="R") if "R" in set(gt.unique()) else (gt=="F")|(gt=="1")
tr["isf"]=isf
sub = tr[~isf & (tr.season>=2023)].copy()
p13=sub.pitcher_team_id==13; b13=sub.batter_team_id==13
tb=sub.top_bottom.astype(str).str.upper().str[0]
h13=np.where(tb=="T",sub.pitcher_team_id,sub.batter_team_id)==13
y=sub.control_success.astype(float)
P("== R행 월별: 리그(비관여) / 13투수측 / 13타자측 / 13홈 / 13원정")
for (s,mo),g in sub.assign(y=y,p13=p13,b13=b13,h13=h13).groupby(["season","game_month"]):
    base=g[~(g.p13|g.b13)].y.mean()
    a=g[g.p13].y.mean(); b=g[g.b13].y.mean()
    hm=g[(g.p13|g.b13)&g.h13].y.mean(); aw=g[(g.p13|g.b13)&~g.h13].y.mean()
    P(f"  {s}-{int(mo):02d}  리그 {base:.3f}  13투 {a-base:+.3f}  13타 {b-base:+.3f}  13홈 {hm-base:+.3f}  13원정 {aw-base:+.3f}  n홈 {int(((g.p13|g.b13)&g.h13).sum())}")
P("== 2021-2024 반기: 13홈 vs 13원정 (R행, 리그 대비)")
sub2 = tr[~isf & (tr.season>=2021)].copy()
p2=sub2.pitcher_team_id==13; b2=sub2.batter_team_id==13
tb2=sub2.top_bottom.astype(str).str.upper().str[0]
h2=np.where(tb2=="T",sub2.pitcher_team_id,sub2.batter_team_id)==13
y2=sub2.control_success.astype(float)
sub2=sub2.assign(y=y2,inv=p2|b2,h=h2,half=np.where(sub2.game_month<=6,"H1","H2"))
for (s,hf),g in sub2.groupby(["season","half"]):
    base=g[~g.inv].y.mean()
    hm=g[g.inv&g.h].y.mean(); aw=g[g.inv&~g.h].y.mean()
    P(f"  {s}{hf}  13홈 {hm-base:+.4f} (n={int((g.inv&g.h).sum())})  13원정 {aw-base:+.4f} (n={int((g.inv&~g.h).sum())})")
