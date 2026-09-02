# -*- coding: utf-8 -*-
"""submit_22 를 조립한다. submit_20(1057) 위에 세 가지를 바꾼다.

    1. 체제 코드 2단계 -> 4단계 교차 (reg4)
    2. 1군 행에서 regular 브랜치 제거, all_reg4 단독
    3. 시드 3개 앙상블 (42 / 1 / 777)
    그리고 온도 보정 제거 (T = 1.0)

체제 코드가 핵심이다
    지금까지 쓰던 2단계는  0 = 옛 퓨처스,  1 = 나머지 전부  였다.
    그러면 **새 퓨처스가 옛 1군과 한 칸**에 들어간다. 그게 1군을 망가뜨리고
    있었다 — fb_arms 에서 체제 처리가 1군을 -18.8 내린 원인이 이것이다.

    4단계 교차로 리그와 시대를 다 분리한다.
        0 = 옛 퓨처스(<=2022)   1 = 옛 1군(<=2022)
        2 = 새 퓨처스(2023~)    3 = 새 1군(2023~)

    브랜치 단독 (관문 VS=2024, 최적 시프트, 시드평균)
                    1군행    퓨처스행
        all_base    871.8     528.1
        all_reg     851.1     609.5     <- 2단계. 1군이 무너진다
        all_reg4    882.8     612.4     <- 양쪽 다 최고
    3단계(옛F / 옛1군 / 새것 전부)는 -4.0 이었다. 시대로만 묶으면 새 퓨처스와
    새 1군이 다시 한 칸이 돼 리그 구분이 사라진다. 교차여야 한다.

관문 (전체 혼합, 최적 절편, 시드별 계산 후 평균)
    submit_20                                895.1   +0.0
    1군 all_base 단독 / 퓨처스 reg2            898.1   +3.0  4/4
    1군 all_reg4 단독 / 퓨처스 reg4 0.4        906.4  +11.3  4/4
                                             시드별 [+9.9, +12.0, +4.6, +18.8]
    3시드 앙상블로는 897.7 -> 907.3 (+9.6)

온도는 뺀다
    순차 1년 전이에서 2023->2024 가 -510.7 이었다. 2024 안에서 맞춰도 +0.0.
    submit_20 의 T=1.0279 는 1에 붙어 있어 해가 없었을 뿐 근거가 없었다.

추론에서 주의
    abs_regime 이 상수가 아니다. 2025 는 전부 새 체제이므로
    퓨처스 행 = 2, 1군 행 = 3. 2단계일 때는 전부 1 이었다.
"""
import io
import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC18 = os.path.join(ROOT, "jaemin_1046_except_MLP.zip")
SRC19 = os.path.join(ROOT, "submit_jaemin_19.zip")
DL = os.path.join(ROOT, "colab", "_dl")
OUT = os.path.join(ROOT, "submit_22")
SEEDS = (42, 1, 777)
SHIFT = -0.0180        # shift_recal: submit_20 대비 D=+0.00351 -> -0.0040 - 4D

HGB_CODE = '''
def predict_hgb(X, z, chunk=4096):
    """HistGB 트리(npz)를 numpy 로 순회해 브랜치별 확률 (n, 3) 을 돌려준다.

    sklearn HistGradientBoostingClassifier 예측을 그대로 재현한다.
      분기 값 <= num_threshold 면 왼쪽 / 결측은 missing_go_to_left
      raw = baseline + 잎값 합,  proba = sigmoid(raw)
    내보낼 때 sklearn 원본과 최대차 5e-16 으로 일치하는 것을 확인했다.
    """
    T, M = z["feat"].shape
    fl = z["feat"].ravel().astype(np.int64)
    tl = z["thr"].ravel()
    ll = z["left"].ravel().astype(np.int64)
    rl = z["right"].ravel().astype(np.int64)
    lfl = z["leaf"].ravel().astype(bool)
    mgl_ = z["mgl"].ravel().astype(bool)
    vl = z["val"].ravel()
    base = z["baselines"]
    depth = int(z["max_depth"])
    n_models = int(z["n_models"])
    per = T // n_models
    root = np.arange(T, dtype=np.int64) * M

    n, Fw = X.shape
    Xf = np.ascontiguousarray(X, dtype=np.float64).ravel()
    out = np.zeros((n, n_models), np.float64)
    off0 = np.repeat(np.arange(chunk, dtype=np.int64) * Fw, T)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        m = e - s
        idx = np.tile(root, m)
        off = off0[:m * T] + s * Fw
        act = np.arange(m * T, dtype=np.int64)
        for _ in range(depth + 1):
            nd = idx[act]
            keep = ~lfl[nd]
            if not keep.any():
                break
            act = act[keep]
            nd = nd[keep]
            v = Xf[off[act] + fl[nd]]
            go_left = np.where(np.isnan(v), mgl_[nd], v <= tl[nd])
            idx[act] = np.where(go_left, ll[nd], rl[nd])
        raw = vl[idx].reshape(m, n_models, per).sum(axis=2)
        out[s:e] = 1.0 / (1.0 + np.exp(-(raw + base[None, :])))
    return out


def route_hgb(pb, is_f, w_branch=0.4):
    """0.6 x all + 0.4 x (경기유형별). 리더보드 989 를 만든 라우팅이다."""
    return (1.0 - w_branch) * pb[:, 0] + w_branch * np.where(is_f, pb[:, 1],
                                                             pb[:, 2])
'''

NEW_INFER = '''        # ---------------- HistGB (CatBoost 와 같은 44열 행렬)
        #
        # 같은 GBDT 총량에서 CatBoost 단독 849.5 -> +HistGB 반반 851.5 (4/4).
        # TabM 과의 상관이 HistGB 0.9031 < CatBoost 0.9492 라 자리가 있다.
        gt = test["game_type"].astype(str).to_numpy()
        is_f_t = gt == "F"
        zh = np.load(resolve("model/hgb.npz"), allow_pickle=False)
        p_hgb = route_hgb(predict_hgb(X.values.astype(np.float64), zh), is_f_t)
        print(f" HistGB   mean={p_hgb.mean():.4f}  "
              f"상관={np.corrcoef(p_cb, p_hgb)[0, 1]:.4f}")

        # ---------------- TabM — 지인 44피처 + abs_regime(4단계)
        #
        #   1군  행: all_reg4 단독
        #   퓨처스행: 0.6 x all_reg4 + 0.4 x futures_reg4
        #   시드 42 / 1 / 777 확률 평균
        #
        # abs_regime 은 여기서 상수가 아니다. 2025 는 전부 새 체제이므로
        # 퓨처스 = 2, 1군 = 3 이다. 학습에서 옛 체제는 0(퓨처스) / 1(1군) 이었다.
        import preprocess as PPF
        with open(resolve("model/history.json"), encoding="utf-8") as f:
            hist_f = PPF.deserialize_history(json.load(f))
        ordered = PPF.sort_by_row_id(test)
        Xf = PPF.build_inference_features(ordered, hist_f)
        isf_o = ordered["game_type"].astype(str).to_numpy() == "F"
        code = np.where(isf_o, 2.0, 3.0)
        Xv = np.c_[Xf.to_numpy(dtype=np.float64), code]

        # 여섯 모델이 같은 전처리 통계를 공유하므로 한 번만 만든다.
        _z0 = np.load(resolve(f"model/r4_all_s{TABM_SEEDS[0]}.npz"),
                      allow_pickle=False)
        _m0 = json.loads(str(_z0["meta"].item()))
        _want = [str(c) for c in _m0["features"]]
        if _want[:-1] != list(Xf.columns) or _want[-1] != "abs_regime":
            raise ValueError("r4 피처가 preprocess 와 어긋난다")
        TN, TC = _fm_prep(Xv, _z0)

        def tabm_avg(branch, rows):
            """브랜치 하나를 시드 평균으로 예측한다. 행마다 독립이라 규칙 4 를 지킨다."""
            acc = None
            for sd in TABM_SEEDS:
                z2 = np.load(resolve(f"model/r4_{branch}_s{sd}.npz"),
                             allow_pickle=False)
                mt2 = json.loads(str(z2["meta"].item()))
                mt2["cat_cardinalities"] = mt2["cards"]
                q = _tabm_forward(TN[rows], TC[rows], mt2, z2, 4096)
                acc = q if acc is None else acc + q
            return acc / len(TABM_SEEDS)

        p_ord = np.zeros(len(Xf), dtype=np.float64)
        rows_r = np.flatnonzero(~isf_o)
        if len(rows_r):
            # regular 브랜치는 안 쓴다. all 학습분의 89%가 이미 1군이라 같은 걸
            # 배우고, 섞으면 희석만 된다 (관문 1군 863.7 -> 867.9).
            p_ord[rows_r] = tabm_avg("all", rows_r)
        rows_f = np.flatnonzero(isf_o)
        if len(rows_f):
            p_ord[rows_f] = (0.6 * tabm_avg("all", rows_f)
                             + 0.4 * tabm_avg("futures", rows_f))

        p_ord = np.clip(p_ord, 0.0, 1.0)
        tm = dict(zip(ordered[ID_COL].tolist(), p_ord))
        p_tabm = np.array([tm[r] for r in test[ID_COL].tolist()], dtype=np.float64)
        print(f" TabM     mean={p_tabm.mean():.4f}  퓨처스 {int(isf_o.sum()):,}행  "
              f"시드 {len(TABM_SEEDS)}개  상관={np.corrcoef(p_cb, p_tabm)[0, 1]:.4f}")

        preds = W_CB * p_cb + W_HGB * p_hgb + W_TABM * p_tabm
        print(f" blended  mean={preds.mean():.4f}  "
              f"({W_CB:.2f}/{W_HGB:.2f}/{W_TABM:.2f})")
        preds = apply_calibration(preds, shift)
        print(f" calibrated mean={preds.mean():.4f}")
'''

NEW_CALIB = '''def apply_calibration(p, shift=CALIB_LOGIT_SHIFT):
    """절편만.  p_out = sigmoid(logit(p) + shift)

    온도(로짓 기울기)는 뺐다. 3폴드로 두 번 확인했다.
      폴드 안에서 맞춘 이득   2022 +3.6 / 2023 +1130.9 / 2024 +0.0
      순차 1년 전이           2022->2023 +107.2 / 2023->2024 -510.7
    2023 만 크게 이득인데 그 해는 퓨처스 체제가 무너진 해라 온도가 붕괴를
    수습한 것뿐이다. 정상 연도에서는 T 가 1 근처고 이득이 없다.

    절편은 이전 제출본과 같은 입력에서 예측평균 차이 D 를 재고 -4D 만큼
    옮겨 정한다(shift_recal.py). 행 단위 단조 변환이라 규칙 4 를 지킨다.
    """
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    z = np.log(p / (1.0 - p)) + shift
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
'''


def grab(src, name):
    lines = src.splitlines()
    s = next(i for i, ln in enumerate(lines) if ln.startswith(f"def {name}("))
    e = s + 1
    while e < len(lines) and not (lines[e] and not lines[e][0].isspace()):
        e += 1
    return "\n".join(lines[s:e]).rstrip() + "\n"


def main():
    z18 = zipfile.ZipFile(SRC18)
    src = z18.read("script.py").decode("utf-8")
    s19 = zipfile.ZipFile(SRC19).read("script.py").decode("utf-8")

    src = src.replace("CALIB_LOGIT_SHIFT = -0.005",
                      f"CALIB_LOGIT_SHIFT = {SHIFT}\n"
                      f"TABM_SEEDS = {tuple(SEEDS)}")
    src = re.sub(r"^W_CB, W_MLP, W_TABM = .*$",
                 "W_CB, W_HGB, W_TABM = 0.1, 0.1, 0.8", src, count=1, flags=re.M)
    src = src.replace(
        "# 3원 앙상블 비중.  최종 = W_CB x CatBoost + W_MLP x flatMLP + W_TABM x TabM",
        "# 3원 앙상블 비중.  최종 = W_CB x CatBoost + W_HGB x HistGB + W_TABM x TabM")
    src = src.replace(
        '    print(f" 앙상블 비중  CatBoost {W_CB:.2f} / flatMLP {W_MLP:.2f} /'
        ' TabM {W_TABM:.2f}")',
        '    print(f" 앙상블 비중  CatBoost {W_CB:.2f} / HistGB {W_HGB:.2f} /'
        ' TabM {W_TABM:.2f}")')

    a = src.index("def apply_calibration(")
    b = src.index("\ndef ", a + 1) + 1
    src = src[:a] + NEW_CALIB + "\n" + src[b:]

    for fn in ("_logit", "_tabm_forward"):
        if f"def {fn}(" not in src:
            src = src.replace("\ndef load_test(",
                              "\n" + grab(s19, fn) + "\n\ndef load_test(", 1)
    src = src.replace("\ndef load_test(", HGB_CODE + "\n\ndef load_test(", 1)

    a = src.index("        # flatMLP")
    b = src.index('        print(f" calibrated mean={preds.mean():.4f}")')
    b = src.index("\n", b) + 1
    src = src[:a] + NEW_INFER + src[b:]

    assert "W_MLP" not in src and "p_mlp" not in src, "flatMLP 잔재가 남았다"
    assert "CALIB_T" not in src, "온도 잔재가 남았다"

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "model"))
    io.open(os.path.join(OUT, "script.py"), "w", encoding="utf-8",
            newline="\n").write(src)
    for n in ("preprocess.py", "requirements.txt"):
        io.open(os.path.join(OUT, n), "wb").write(z18.read(n))
    for n in ("trees.npz", "history.json", "config.json"):
        io.open(os.path.join(OUT, "model", n), "wb").write(z18.read("model/" + n))
    shutil.copy(os.path.join(DL, "hgb.npz"), os.path.join(OUT, "model", "hgb.npz"))
    for br in ("all", "futures"):
        for sd in SEEDS:
            f = os.path.join(DL, f"r4_{br}_s{sd}.npz")
            if not os.path.exists(f):
                raise SystemExit(f"없음: {f} — 서버에서 내려받아라")
            shutil.copy(f, os.path.join(OUT, "model", f"r4_{br}_s{sd}.npz"))

    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(OUT) for f in fs)
    print(f"  {OUT}  {tot/1e6:.1f}MB   시드 {SEEDS}   shift={SHIFT}")
    for r, _, fs in os.walk(OUT):
        for f in sorted(fs):
            p = os.path.join(r, f)
            print(f"    {os.path.relpath(p, OUT):34s} {os.path.getsize(p)/1e6:7.2f}MB")


if __name__ == "__main__":
    main()
