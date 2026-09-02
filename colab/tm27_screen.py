# -*- coding: utf-8 -*-
"""팀원 사양서대로 트랙맨 27피처를 만들고 편상관으로 거른다. 로컬 CPU.

왜 앞선 기각이 이걸 안 덮나
    내가 잰 트랙맨 후보는 cluster_v5 라벨과 max_speed / fb_hb_norm / fb_ivb /
    n_pitch_types / n_breaking_types 였다. 이 사양의 27개 중 **둘이 완전히 새롭다**.

        fb_rate_gap    카운트 조건부 구종 선택. 불리할 때 - 유리할 때 패스트볼 비율.
                       우리 44열의 asof_pitcher_fastball_rate 는 전체 평균뿐이라
                       이 축이 아예 없다. 사양서가 27개 중 제구와 편상관이
                       가장 강하다고 적은 항목이다 (r = -0.28).
        resid_s_*      구종 효과를 뺀 뒤의 릴리스 산포. 2단계 집계라
                       '같은 구종 안에서 얼마나 반복 가능한가' 를 잰다.
                       기전이 제구와 직결된다.

    커버리지도 다르다. 내 fbshape 검사는 v5 클러스터가 2022~2024 뿐이라
    엄격한 as-of 로 학습 14.2% 였는데, 이 사양은 기간 내 전 투구를 한 덩어리로
    집계해 443명 중 411명이 <=2023 프로필을 갖는다.

남는 의심 — 이건 사양서가 안 다룬다
    27개 전부 **투수 상수**다. 한 투수의 모든 행에 같은 값이 붙는다.
    TabM 에는 793칸 pitcher_id 임베딩이 있어 같은 정보를 이미 담을 수 있다.
    그리고 CatBoost 는 pitcher_id 를 숫자 열로 받아 못 쓰므로, 트리로 재면
    투수 단위 피처가 항상 좋아 보인다. 오늘 직접 봤다 —
        lg_f_share  CatBoost 3폴드 +12.2/+77.8/+3.5  vs  TabM -27.5

    그래서 여기서는 44열(pitcher_id 포함)을 통제한 **편상관**으로 잰다.
    통제 후에도 남으면 임베딩이 못 담은 정보라는 뜻이다.

    그리고 저표본 투수에서 더 클 것이라는 가설도 같이 본다 — 임베딩은 라벨로
    배우므로 표본이 적으면 부실하고, 트랙맨 프로필은 수천 구로 측정된 값이다.

기간 규약 (사양서 그대로)
    MAX_SEASON = 2023   VS=2024 관문용. 검증 연도가 프로필에 안 들어간다.
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = "open (1)/data"
MAP = os.path.join(ROOT, "trackman_map", "pitcher_map_full.csv")
VS = 2024
MAX_SEASON = VS - 1
THRESH = 0.005

import features44 as F                                          # noqa: E402

PT_MAP = {"fastball": "fb", "fourseam": "fb", "fourseamfastball": "fb",
          "sinker": "sinker", "twoseam": "sinker", "cutter": "cutter",
          "slider": "slider", "curveball": "curveball", "curve": "curveball",
          "changeup": "changeup", "splitter": "splitter"}
MIX = ["mix_fb", "mix_sinker", "mix_cutter", "mix_slider", "mix_curveball",
       "mix_changeup", "mix_splitter", "mix_misc"]
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side"]


def build_profile():
    mp = pd.read_csv(MAP, encoding="utf-8-sig")
    mp = mp[mp["status"] == "채택"][["pitcher_id", "trackman_id"]]
    mp = mp.drop_duplicates("pitcher_id")
    # 필요한 열만 읽는다. 전체 30열을 올리면 1.7M행에서 메모리가 터진다.
    need = ["pitcher_trackman_id", "season", "auto_pitch_type",
            "pitch_type_group", "balls_before", "strikes_before"] + PHYS
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"),
                     encoding="utf-8-sig", usecols=need)
    keep = ((tm["season"] <= MAX_SEASON)
            & tm["pitcher_trackman_id"].isin(set(mp["trackman_id"])))
    tm = tm.loc[keep].reset_index(drop=True)
    for c in PHYS:
        tm[c] = pd.to_numeric(tm[c], errors="coerce")
    g = "pitcher_trackman_id"

    # (1) 구종 비율 8개. Fastball 과 FourSeam 은 같은 구종의 연도별 표기다.
    key = (tm["auto_pitch_type"].astype(str).str.lower()
           .str.replace(r"[\s\-_]", "", regex=True).map(PT_MAP).fillna("misc"))
    tm["_pt"] = key
    cnt = tm.groupby([g, "_pt"]).size().unstack(fill_value=0)
    for c in ("fb", "sinker", "cutter", "slider", "curveball", "changeup",
              "splitter", "misc"):
        if c not in cnt:
            cnt[c] = 0
    mix = cnt[["fb", "sinker", "cutter", "slider", "curveball", "changeup",
               "splitter", "misc"]].div(cnt.sum(1), axis=0)
    mix.columns = MIX

    # (2) 패스트볼 물리 7개. 필터는 pitch_type_group == 'fastball' (싱커 포함)
    fbm = tm[tm["pitch_type_group"].astype(str).str.lower() == "fastball"]
    fb = fbm.groupby(g)[PHYS].mean()
    fb.columns = [f"fb_m_{c}" for c in PHYS]

    # (3) 릴리스 산포 3개. 구종 효과를 뺀 잔차의 표준편차.
    rs = {}
    for c in ("rel_height", "rel_side", "rel_speed"):
        mu = tm.groupby([g, "auto_pitch_type"])[c].transform("mean")
        rs[f"resid_s_{c}"] = (tm[c] - mu).groupby(tm[g]).std()
    rs = pd.DataFrame(rs)

    # (4) 아스널 구조 3개 — 비율에서 파생
    ar = pd.DataFrame(index=mix.index)
    ar["arsenal_n"] = (mix > 0.03).sum(1)
    p = mix.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(p > 0, np.log(p), 0.0)
    ar["arsenal_entropy"] = -(p * lg).sum(1)
    ar["primary_share"] = mix.max(1)

    # (5) 구종 선택 성향 3개 — 카운트 조건부
    isfb = (tm["pitch_type_group"].astype(str).str.lower() == "fastball")
    ah = tm["strikes_before"] > tm["balls_before"]
    bh = tm["balls_before"] > tm["strikes_before"]
    sel = pd.DataFrame({
        "fb_rate_ahead": isfb[ah].groupby(tm[g][ah]).mean(),
        "fb_rate_behind": isfb[bh].groupby(tm[g][bh]).mean()})
    sel["fb_rate_gap"] = sel["fb_rate_behind"] - sel["fb_rate_ahead"]

    prof = (mix.join(fb, how="outer").join(rs, how="outer")
            .join(ar, how="outer").join(sel, how="outer"))
    prof["tm_n"] = tm.groupby(g).size()
    prof = prof.reset_index().rename(columns={g: "trackman_id"})
    return prof.merge(mp, on="trackman_id", how="inner")


def resid(x, C):
    return x - C @ np.linalg.lstsq(C, x, rcond=None)[0]


def partial(x, y, C):
    rx, ry = resid(x, C), resid(y, C)
    sx, sy = rx.std(), ry.std()
    return 0.0 if sx < 1e-12 or sy < 1e-12 else float((rx * ry).mean() / (sx * sy))


if __name__ == "__main__":
    t0 = time.time()
    prof = build_profile()
    feats = [c for c in prof.columns if c not in ("trackman_id", "pitcher_id")]
    print(f"  프로필 투수 {len(prof)}명  피처 {len(feats)}개  "
          f"MAX_SEASON={MAX_SEASON}  {time.time()-t0:.0f}s")
    print(f"  검산  mix 합 {prof[MIX].sum(1).mean():.6f}  "
          f"fb_m_rel_speed 평균 {prof['fb_m_rel_speed'].mean():.1f}  "
          f"arsenal_entropy {prof['arsenal_entropy'].mean():.3f}  "
          f"fb_rate_gap {prof['fb_rate_gap'].mean():.3f}")

    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id", "season", "game_type", "pitcher_id",
                               "asof_pitcher_n"])
    m = raw.merge(prof, on="pitcher_id", how="left")
    d = F.build(DATA, VS=VS)
    X44 = d["X44"].astype(np.float64)
    y = d["y"].astype(np.float64)
    m_tr, isf = d["m_tr"], d["is_f"]
    assert len(m) == len(X44)

    med = np.nanmedian(X44[m_tr], 0)
    Xf = np.where(np.isnan(X44), med, X44)
    ok = Xf[m_tr].std(0) > 1e-9
    hit = m["fb_m_rel_speed"].notna().to_numpy()
    lown = (raw["asof_pitcher_n"].to_numpy() < 500)     # 저표본 투수

    subs = {"전체": m_tr, "매칭행": m_tr & hit, "매칭+퓨처스": m_tr & hit & isf,
            "매칭+저표본": m_tr & hit & lown}
    C = {k: np.c_[np.ones(v.sum()), Xf[v][:, ok]] for k, v in subs.items()}
    print(f"\n  행 커버리지  전체 대비 {hit[m_tr].mean()*100:.1f}%   "
          f"퓨처스 {hit[m_tr & isf].mean()*100:.1f}%   "
          f"저표본(asof_n<500) {hit[m_tr & lown].mean()*100:.1f}%")
    print(f"  기준선 |r| > {THRESH}   (basic 후보 12개는 전부 <0.0025, "
          f"통과한 lg_f_share 는 -0.0174)\n")
    print(f"  {'후보':22s} " + " ".join(f"{k:>12s}" for k in subs) + "  판정")

    keep = []
    for c in feats:
        v = m[c].to_numpy(np.float64)
        row, best = [], 0.0
        for k, msk in subs.items():
            vv = v[msk]
            if np.isnan(vv).all():
                row.append(0.0)
                continue
            vv = np.where(np.isnan(vv), np.nanmedian(vv), vv)
            r = partial(vv, y[msk], C[k])
            row.append(r)
            best = max(best, abs(r))
        good = best > THRESH
        if good:
            keep.append(c)
        print(f"  {c:22s} " + " ".join(f"{x:+12.4f}" for x in row)
              + f"  {'통과' if good else ''}")

    print(f"\n  통과 {len(keep)}개: {keep if keep else '없음'}   "
          f"{time.time()-t0:.0f}s")
    print("  27개 전부 투수 상수다. 44열(pitcher_id 포함)을 통제한 뒤 남은 값이므로")
    print("  임베딩이 못 담은 정보만 보인다. '저표본' 열이 큰 항목이 특히 유망하다 —")
    print("  임베딩은 라벨로 배워서 표본이 적으면 부실한데 프로필은 측정값이다.")
    if keep:
        prof[["pitcher_id"] + keep].to_csv(
            os.path.join(SC, "_dl", "tm27_pass.csv.gz"), index=False,
            compression="gzip")
        print("  통과분 저장: colab/_dl/tm27_pass.csv.gz")
