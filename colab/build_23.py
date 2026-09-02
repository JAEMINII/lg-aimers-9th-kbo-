# -*- coding: utf-8 -*-
"""submit_20 을 조립한다. submit_18(1048) 위에 세 가지를 얹는다.

    1. HistGB 를 앙상블에 추가          0.1 CB / 0.1 HistGB / 0.8 TabM
    2. 퓨처스 행에만 regime TabM        1군 행은 submit_18 그대로
    3. 온도 보정                        sigmoid(T x logit(p) + c)

근거 (관문 VS=2024, 전체 R+F, TabM 1시드 = 배치와 같은 구성)
    퓨처스만 regime      TabM 864.0 -> 879.2   퓨처스 행만 492.5 -> 630.1
    0.1/0.1/0.8          843.2 -> 851.5  (4/4 시드 양수)
    온도                 T=1.0279 이고 이득 +0.6. 1에 붙어 있어 사실상 중립이다.
                         다년 전이 시험에서 큰 T 는 세 폴드 중 둘에서 크게 마이너스라
                         (2022 -744, 2024 -144) 관문에서 나온 값을 그대로만 쓴다.

    절편은 여기서 정하지 않는다. submit_18 과 같은 입력에 돌려 예측평균 차이 D 를
    재고 -0.005 - 4D 로 맞춘다(shift_recal.py). submit_19 는 그걸 안 해서
    10~20점을 잃었다.

파일 구성
    script.py            submit_18 것에 HistGB 순회 / regime TabM / 새 라우팅을 얹음
    preprocess.py        지인 전처리 (그대로)
    model/trees.npz      CatBoost (그대로)
    model/history.json   지인 history (그대로)
    model/{all,regular}_tabm_seed_42.npz    base TabM — 1군 행에 쓴다
    model/fbregime_{all,futures}_seed42.npz regime TabM — 퓨처스 행에 쓴다
    model/hgb.npz        HistGB 3브랜치

    base 의 futures_tabm_seed_42.npz 는 뺀다. 퓨처스 행을 regime 이 맡으므로
    쓸 데가 없다. 12MB 중 3.4MB 가 줄고 추론 패스도 하나 준다.
"""
import io
import json
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC18 = os.path.join(ROOT, "jaemin_1046_except_MLP.zip")
SRC19 = os.path.join(ROOT, "submit_jaemin_19.zip")
DL = os.path.join(ROOT, "colab", "_dl")
OUT = os.path.join(ROOT, "submit_23")
CALIB_T = 1.0279
SHIFT = -0.0240        # 시프트 A/B 용. submit_20 과 이 상수 하나만 다르다.


def grab(src, name):
    """submit_19 script.py 에서 함수 하나를 통째로 떼온다."""
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"def {name}("))
    end = start + 1
    while end < len(lines) and not (lines[end] and not lines[end][0].isspace()):
        end += 1
    return "\n".join(lines[start:end]).rstrip() + "\n"


HGB_CODE = '''
def predict_hgb(X, z, chunk=4096):
    """HistGB 트리(npz)를 numpy 로 순회해 브랜치별 확률 (n, 3) 을 돌려준다.

    sklearn HistGradientBoostingClassifier 의 예측을 그대로 재현한다.
      분기    값 <= num_threshold 면 왼쪽
      결측    missing_go_to_left 플래그
      raw     baseline + 잎값 합,  proba = sigmoid(raw)
    내보낼 때 sklearn 원본과 최대차 5e-16 으로 일치하는 것을 확인했다.

    노드 배열을 1차원으로 펴서 1200그루를 한 번에 순회하고, 매 단계
    '아직 잎에 안 닿은' 위치만 남긴다. submit_jaemin_2(958) 에서 쓰던 방식이다.
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

NEW_INFER = '''        # ---------------- HistGB (CatBoost 와 같은 44열 행렬을 쓴다)
        #
        # 리더보드에서 HistGB 60+40 = 989, CatBoost 60+40 = 987 로 동점이었다.
        # 둘은 같은 계열이라 서로 대체재에 가깝지만, TabM 과의 상관이
        # HistGB 0.9031 < CatBoost 0.9492 로 더 떨어져 있어 자리가 있다.
        # 시즌가중은 2.0 을 쓴다 — 단독은 1.0 이 높지만(869.9 vs 854.1)
        # 2.0 쪽이 TabM 과 덜 겹쳐 혼합에서는 더 낫다(851.5 vs 848.9).
        gt = test["game_type"].astype(str).to_numpy()
        is_f_t = gt == "F"
        zh = np.load(resolve("model/hgb.npz"), allow_pickle=False)
        p_hgb = route_hgb(predict_hgb(X.values.astype(np.float64), zh), is_f_t)
        print(f" HistGB   mean={p_hgb.mean():.4f}  "
              f"상관={np.corrcoef(p_cb, p_hgb)[0, 1]:.4f}")

        # ---------------- TabM — 지인 44피처
        #
        # 라우팅을 행 종류로 나눈다.
        #   1군  행: base 3브랜치  0.6 x all + 0.4 x regular   (submit_18 그대로)
        #   퓨처스행: regime 판본  0.6 x all + 0.4 x futures
        #
        # 체제 처리는 퓨처스를 크게 올리고 1군을 내린다(관문 4시드 짝비교:
        # 퓨처스 +157.6 t=12.4 4/4, 1군 -18.8 0/4). 퓨처스가 11.8% 뿐이라
        # 전체에 걸면 희석돼 +4.7 로 무의미해진다. 그래서 퓨처스에만 건다.
        # 관문에서 TabM 864.0 -> 879.2, 퓨처스 행만 492.5 -> 630.1.
        import preprocess as PPF
        with open(resolve("model/history.json"), encoding="utf-8") as f:
            hist_f = PPF.deserialize_history(json.load(f))
        ordered = PPF.sort_by_row_id(test)
        Xf = PPF.build_inference_features(ordered, hist_f)
        isf_o = ordered["game_type"].astype(str).to_numpy() == "F"

        meta_a, arc_a = load_bundle(resolve("model/all_tabm_seed_42.npz"))
        p_ord = predict_frame(Xf, meta_a, arc_a, chunk_size=4096)

        rows_r = np.flatnonzero(~isf_o)
        if len(rows_r):
            mt, ar = load_bundle(resolve("model/regular_tabm_seed_42.npz"))
            pr = predict_frame(Xf.iloc[rows_r], mt, ar, chunk_size=4096)
            p_ord[rows_r] = 0.6 * p_ord[rows_r] + 0.4 * pr

        rows_f = np.flatnonzero(isf_o)
        if len(rows_f):
            # abs_regime 은 추론에서 항상 1 이다. 2025 는 전부 새 체제다.
            Xv = np.c_[Xf.to_numpy(dtype=np.float64), np.ones(len(Xf))]
            acc = {}
            for br in ("all", "futures"):
                z2 = np.load(resolve(f"model/fbregime_{br}_seed42.npz"),
                             allow_pickle=False)
                mt2 = json.loads(str(z2["meta"].item()))
                mt2["cat_cardinalities"] = mt2["cards"]
                want = [str(c) for c in mt2["features"]]
                if want[:-1] != list(Xf.columns) or want[-1] != "abs_regime":
                    raise ValueError("fbregime 피처가 preprocess 와 어긋난다")
                Tn, Tc = _fm_prep(Xv[rows_f], z2)
                acc[br] = _tabm_forward(Tn, Tc, mt2, z2, 4096)
            p_ord[rows_f] = 0.6 * acc["all"] + 0.4 * acc["futures"]

        p_ord = np.clip(p_ord, 0.0, 1.0)
        tm = dict(zip(ordered[ID_COL].tolist(), p_ord))
        p_tabm = np.array([tm[r] for r in test[ID_COL].tolist()], dtype=np.float64)
        print(f" TabM     mean={p_tabm.mean():.4f}  퓨처스 {int(is_f_t.sum()):,}행  "
              f"상관={np.corrcoef(p_cb, p_tabm)[0, 1]:.4f}")

        preds = W_CB * p_cb + W_HGB * p_hgb + W_TABM * p_tabm
        print(f" blended  mean={preds.mean():.4f}  "
              f"({W_CB:.2f}/{W_HGB:.2f}/{W_TABM:.2f})")
        preds = apply_calibration(preds, shift)
        print(f" calibrated mean={preds.mean():.4f}  T={CALIB_T}")
'''

NEW_CALIB = '''def apply_calibration(p, shift=CALIB_LOGIT_SHIFT, T=None):
    """온도 + 절편.  p_out = sigmoid(T x logit(p) + shift)

    T 는 관문(2019~2023 학습 -> 2024)에서 이 배치 구성 그대로 맞춘 값이다.
    1 에 거의 붙어 있고 이득도 +0.6 이라 사실상 중립이다. 크게 잡지 않는 이유는
    다년 전이 시험 때문이다 — 다른 폴드에서 맞춘 T 를 가져다 쓰면 2022 에서
    -744, 2024 에서 -144 였다. 온도는 그 해에만 맞는 값이다.

    행 단위 단조 변환이라 순위는 보존되고 규칙 4 도 지킨다.
    """
    T = CALIB_T if T is None else T
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    z = T * np.log(p / (1.0 - p)) + shift
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
'''


def main():
    s18 = zipfile.ZipFile(SRC18)
    src = s18.read("script.py").decode("utf-8")
    s19 = zipfile.ZipFile(SRC19).read("script.py").decode("utf-8")

    # 1) 상수
    src = src.replace("CALIB_LOGIT_SHIFT = -0.005",
                      f"CALIB_LOGIT_SHIFT = {SHIFT}\nCALIB_T = {CALIB_T}")
    src = re.sub(r"^W_CB, W_MLP, W_TABM = .*$",
                 f"W_CB, W_HGB, W_TABM = 0.1, 0.1, 0.8",
                 src, count=1, flags=re.M)
    assert "W_CB, W_HGB, W_TABM" in src, "비중 줄을 못 찾았다"

    # 2) 보정 함수 교체
    old_cal_start = src.index("def apply_calibration(")
    old_cal_end = src.index("\ndef ", old_cal_start + 1) + 1
    src = src[:old_cal_start] + NEW_CALIB + "\n" + src[old_cal_end:]

    # 3) submit_19 에만 있는 함수 두 개를 가져온다
    for fn in ("_logit", "_tabm_forward"):
        if f"def {fn}(" not in src:
            src = src.replace("\ndef load_test(", "\n" + grab(s19, fn)
                              + "\n\ndef load_test(", 1)

    # 4) HistGB 순회
    src = src.replace("\ndef load_test(", HGB_CODE + "\n\ndef load_test(", 1)

    # 5) 추론부 교체. flatMLP 블록부터 잘라낸다 — submit_18 에서 이미 비중 0 이라
    #    죽은 코드였고, W_MLP 상수를 없앴으므로 남겨 두면 NameError 가 난다.
    src = src.replace(
        "# 3원 앙상블 비중.  최종 = W_CB x CatBoost + W_MLP x flatMLP + W_TABM x TabM",
        "# 3원 앙상블 비중.  최종 = W_CB x CatBoost + W_HGB x HistGB + W_TABM x TabM")
    src = src.replace(
        '    print(f" 앙상블 비중  CatBoost {W_CB:.2f} / flatMLP {W_MLP:.2f} /'
        ' TabM {W_TABM:.2f}")',
        '    print(f" 앙상블 비중  CatBoost {W_CB:.2f} / HistGB {W_HGB:.2f} /'
        ' TabM {W_TABM:.2f}")')
    a = src.index("        # flatMLP")
    b = src.index('        print(f" calibrated mean={preds.mean():.4f}")')
    b = src.index("\n", b) + 1
    src = src[:a] + NEW_INFER + src[b:]

    # 6) flatMLP 잔재 정리 — W_MLP 를 지웠으므로 참조가 남으면 죽는다
    assert "W_MLP" not in src, "W_MLP 참조가 남았다:\n" + \
        "\n".join(ln for ln in src.splitlines() if "W_MLP" in ln)
    assert "p_mlp" not in src, "p_mlp 참조가 남았다"

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "model"))
    io.open(os.path.join(OUT, "script.py"), "w", encoding="utf-8",
            newline="\n").write(src)
    for n in ("preprocess.py", "requirements.txt"):
        io.open(os.path.join(OUT, n), "wb").write(s18.read(n))
    for n in ("trees.npz", "history.json", "config.json",
              "all_tabm_seed_42.npz", "regular_tabm_seed_42.npz"):
        io.open(os.path.join(OUT, "model", n), "wb").write(s18.read("model/" + n))
    shutil.copy(os.path.join(DL, "hgb.npz"), os.path.join(OUT, "model", "hgb.npz"))
    for br in ("all", "futures"):
        f = os.path.join(DL, f"fbregime_{br}_seed42.npz")
        if not os.path.exists(f):
            raise SystemExit(f"없음: {f} — 서버에서 내려받아라")
        shutil.copy(f, os.path.join(OUT, "model", f"fbregime_{br}_seed42.npz"))

    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(OUT) for f in fs)
    print(f"  {OUT}  파일 {sum(len(fs) for _,_,fs in os.walk(OUT))}개  "
          f"{tot/1e6:.1f}MB")
    for r, _, fs in os.walk(OUT):
        for f in sorted(fs):
            p = os.path.join(r, f)
            print(f"    {os.path.relpath(p, OUT):40s} {os.path.getsize(p)/1e6:7.2f}MB")


if __name__ == "__main__":
    main()
