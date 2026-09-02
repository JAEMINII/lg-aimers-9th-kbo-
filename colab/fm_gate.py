# -*- coding: utf-8 -*-
"""FM — 강규제 저랭크 2차 상호작용 전용 (딥타워 없음). 조언 라운드3 1순위.

핵심: pitcher_id/batter_id 는 1차(bias) 제외, 상호작용에만 참여 —
ID 를 '수준 암기'가 아니라 '관계 효과'로만 쓰게 한다.
저장: fm_{VS}_s{sd}.npy. 판정은 로컬 54-믹스 블렌드 3폴드."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = "/root/aimers"
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
DEV = torch.device("cuda")
DIM = int(os.environ.get("FM_DIM", "8"))
EPOCHS = int(os.environ.get("FM_EP", "2"))
LR = 1e-2
WD_V = float(os.environ.get("FM_WD", "3e-4"))
BS = 16384
SEEDS = (42, 1)

tr = pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig")
rid = tr["row_id"].astype(str).str.extract(r"(\d+)$", expand=False).astype("int64")
tr = tr.assign(_rid=rid).sort_values("_rid", kind="mergesort").reset_index(drop=True)
y_all = tr["control_success"].to_numpy(np.float32)
season = tr["season"].to_numpy()
isf_all = tr["game_type"].astype(str).to_numpy() == "F"


def build_fields(df, mu, edges=None, fit_mask=None):
    """필드 리스트: (이름, 코드배열, 카디널리티, 1차참여여부)"""
    KAP = 300.0
    n = df["asof_pitcher_n"].fillna(0.).to_numpy(float)
    r = df["asof_pitcher_success_rate"].fillna(mu).to_numpy(float)
    rs = (n * r + KAP * mu) / (n + KAP)
    nb = df["asof_batter_n"].fillna(0.).to_numpy(float)
    rb = df["asof_batter_success_rate"].fillna(mu).to_numpy(float)
    rbs = (nb * rb + KAP * mu) / (nb + KAP)
    cnt = df["balls_before"].to_numpy(int) * 3 + df["strikes_before"].to_numpy(int)
    hp = (df["pitcher_hand"].to_numpy(int) - 1) * 2 + (df["batter_hand"].to_numpy(int) - 1)
    base = (df["runner_on_1b"].to_numpy(int) + 2 * df["runner_on_2b"].to_numpy(int)
            + 4 * df["runner_on_3b"].to_numpy(int))
    nums = {"p_rs": rs, "b_rs": rbs,
            "p_ball": df["asof_pitcher_ball_rate"].to_numpy(float),
            "p_strk": df["asof_pitcher_strike_rate"].to_numpy(float),
            "li": df["li"].to_numpy(float),
            "logn": np.log1p(n), "lognb": np.log1p(nb)}
    ed = {} if edges is None else edges
    fields = []
    for k, v in nums.items():
        if edges is None:
            qs = np.quantile(v[fit_mask][np.isfinite(v[fit_mask])],
                             np.linspace(0, 1, 17)[1:-1])
            ed[k] = np.unique(qs)
        b = np.digitize(np.where(np.isfinite(v), v, np.inf), ed[k])
        b = np.where(np.isfinite(v), b + 1, 0)
        fields.append((k, b.astype(np.int64), len(ed[k]) + 3, True))
    cats = {"cnt": (cnt, 12), "hp": (hp, 4),
            "outs": (df["outs_before"].to_numpy(int), 3),
            "base": (base, 8),
            "inning": (np.clip(df["inning"].to_numpy(int), 1, 12) - 1, 12),
            "month": (df["game_month"].to_numpy(int) - 1, 12),
            "gt": ((df["game_type"].astype(str).to_numpy() == "F").astype(int), 2),
            "nrun": (np.clip(df["num_runners_on"].to_numpy(int), 0, 3), 4)}
    for k, (v, c) in cats.items():
        fields.append((k, np.clip(v, 0, c - 1).astype(np.int64), c, True))
    for k, col in (("pid", "pitcher_id"), ("bid", "batter_id")):
        v = df[col].to_numpy()
        if edges is None:
            ed["m_" + k] = {int(x): i + 1 for i, x in enumerate(np.unique(v[fit_mask]))}
        mp = ed["m_" + k]
        fields.append((k, np.array([mp.get(int(x), 0) for x in v], np.int64),
                       max(mp.values()) + 1, False))          # 1차 제외
    return fields, ed


class FM(nn.Module):
    def __init__(self, fields, dim):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, dim) for _, _, c, _ in fields])
        self.lin = nn.ModuleList([nn.Embedding(c, 1) if fo else None
                                  for _, _, c, fo in fields])
        self.b = nn.Parameter(torch.zeros(1))
        for e in self.emb:
            nn.init.normal_(e.weight, 0, 0.01)
        for l in self.lin:
            if l is not None:
                nn.init.zeros_(l.weight)

    def forward(self, X):
        vs = torch.stack([e(X[:, i]) for i, e in enumerate(self.emb)], 1)
        s = vs.sum(1)
        inter = 0.5 * ((s * s).sum(1) - (vs * vs).sum(2).sum(1))
        lin = sum(l(X[:, i]).squeeze(1) for i, l in enumerate(self.lin)
                  if l is not None)
        return self.b + lin + inter


for VS in (2024, 2023, 2022):
    m_tr = season < VS
    m_va = season == VS
    mu = float(y_all[m_tr].mean())
    fields, ed = build_fields(tr, mu, edges=None, fit_mask=m_tr)
    X = np.column_stack([f[1] for f in fields])
    w = np.where(isf_all & (season <= 2022), 0.1, 1.0) \
        * 1.5 ** (season.astype(np.float32) - 2019.0)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(y_all)
    Wt = torch.from_numpy(w.astype(np.float32))
    tri = np.flatnonzero(m_tr)
    vai = np.flatnonzero(m_va)
    for sd in SEEDS:
        torch.manual_seed(sd)
        np.random.seed(sd)
        t0 = time.time()
        m = FM(fields, DIM).to(DEV)
        opt = torch.optim.AdamW([
            {"params": [p for e in m.emb for p in e.parameters()],
             "weight_decay": WD_V},
            {"params": [p for l in m.lin if l is not None
                        for p in l.parameters()] + [m.b], "weight_decay": 1e-6},
        ], lr=LR)
        for ep in range(EPOCHS):
            perm = np.random.permutation(tri)
            for a in range(0, len(perm), BS):
                bidx = torch.from_numpy(perm[a:a + BS])
                xb = Xt[bidx].to(DEV)
                z = m(xb)
                per = nn.functional.binary_cross_entropy_with_logits(
                    z, Yt[bidx].to(DEV), reduction="none")
                wb = Wt[bidx].to(DEV)
                loss = (per * wb).sum() / wb.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()
        out = []
        with torch.no_grad():
            for a in range(0, len(vai), 65536):
                xb = Xt[torch.from_numpy(vai[a:a + 65536])].to(DEV)
                out.append(torch.sigmoid(m(xb)).cpu().numpy())
        p = np.concatenate(out)
        tag = os.environ.get("FM_TAG", "")
        np.save(DL + f"/fm{tag}_{VS}_s{sd}.npy", p)
        print(f"  fm VS{VS} s{sd}  {time.time()-t0:.0f}s  평균 {p.mean():.4f}",
              flush=True)
        del m
        torch.cuda.empty_cache()
print("FM 끝", flush=True)
