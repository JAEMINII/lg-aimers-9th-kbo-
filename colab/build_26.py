# -*- coding: utf-8 -*-
"""submit_26 을 조립한다. submit_20(1057)에서 **퓨처스 경로만** 갈아끼운다.

왜 1군을 안 건드리나
    우리 학습 코드 계열이 1052 에서 막혀 있다.
        지인 1군   submit_15 1046  submit_18 1048  submit_20 1057
        우리 1군   submit_19 1020  submit_21 1024  submit_23 1048
                   submit_24 1049  submit_25 1052
    스케줄러 결함을 찾아 고쳐서 +21.6 을 회수했는데도(21 -> 23) 아직 5점이 모자란다.
    두 코드 사이에 못 찾은 차이가 하나 더 남아 있다.

    그래서 1군(88.2%)은 submit_20 파일을 **바이트 단위로** 그대로 쓰고,
    오늘 확보한 이득은 전부 퓨처스(11.8%)에 싣는다.

바뀌는 것 — 퓨처스 경로 모델 두 종류뿐
    submit_20      fbregime_{all,futures}_seed42   구코드, 망가진 스케줄러,
                                                   2단계 코드, lr 2e-3, 1시드,
                                                   Stage2 1에폭
    submit_26      r4_{all,futures}_s{42,1,777}    에폭단위 스케줄러, 4단계 코드,
                                                   lr 3e-3, 3시드,
                                                   Stage2 all 1에폭 / futures 4에폭

관문 (퓨처스 구간, 구간 분모, 최적 시프트)
    submit_20 현행 (구코드 reg2, 1시드)      625.8
    고친코드 reg4 + Stage2 e4, 1시드         660.5   +34.7  4/4
    고친코드 reg4 + Stage2 e4, 3시드         666.2   +40.4
    퓨처스가 11.8% 이므로 전체 환산 약 +4.8.

Stage2 에폭을 4로 한 근거
    퓨처스 Stage2 는 2024 표본이 30,010행 = 15 스텝뿐이다.
    e1 650.2 / e2 655.8 / e3 659.1 / e4 660.5 / e5 661.0 / e6 661.4 /
    e7 660.9 / e8 660.6 — e4~e8 이 고원이고 그 안의 차이는 SE(2~3) 안쪽이다.
    시즌가중을 3~8 고원에서 3.5 로 고른 것과 같은 규율로 덜 극단적인 e4 를 쓴다.
    1군/all 브랜치는 223,497행(110 스텝)이라 1에폭으로 충분하다(e2 +0.6, 2/4).

추론 비용
    1군 행은 submit_20 그대로 2패스. 퓨처스 행만 3시드 x 2브랜치 = 6패스인데
    그 구간이 11.8% 라 전체는 2.0 -> 2.47 패스등가, 약 1.24배다.
"""
import io
import json
import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC20 = os.path.join(ROOT, "submit_jaemin_20.zip")
SRC19 = os.path.join(ROOT, "submit_jaemin_19.zip")
DL = os.path.join(ROOT, "colab", "_dl", "f26")
OUT = os.path.join(ROOT, "submit_26")
SEEDS = (42, 1, 777)
SHIFT = -0.0038        # 앵커 -0.0070(submit_20 계열 최적) - 4 x D(-0.00079)

NEW_FUT = '''        # ---------------- 퓨처스 경로 (여기만 submit_20 에서 바뀐다)
        #
        #   0.6 x all_reg4 + 0.4 x futures_reg4,  시드 42/1/777 확률 평균
        #
        # abs_regime 은 4단계 교차다. 추론에서 2025 는 전부 새 체제이므로
        # 퓨처스 = 2, 1군 = 3. (2단계일 때는 전부 1 이었다.)
        # 여기서는 퓨처스 행만 계산하므로 코드는 2 하나뿐이다.
        rows_f = np.flatnonzero(isf_o)
        if len(rows_f):
            Xv = np.c_[Xf.to_numpy(dtype=np.float64),
                       np.full(len(Xf), 2.0)]
            _z0 = np.load(resolve(f"model/r4_all_s{TABM_SEEDS[0]}.npz"),
                          allow_pickle=False)
            _m0 = json.loads(str(_z0["meta"].item()))
            _want = [str(c) for c in _m0["features"]]
            if _want[:-1] != list(Xf.columns) or _want[-1] != "abs_regime":
                raise ValueError("r4 피처가 preprocess 와 어긋난다")
            TN, TC = _fm_prep(Xv, _z0)

            def r4_avg(branch):
                acc = None
                for sd in TABM_SEEDS:
                    z2 = np.load(resolve(f"model/r4_{branch}_s{sd}.npz"),
                                 allow_pickle=False)
                    mt2 = json.loads(str(z2["meta"].item()))
                    mt2["cat_cardinalities"] = mt2["cards"]
                    q = _tabm_forward(TN[rows_f], TC[rows_f], mt2, z2, 4096)
                    acc = q if acc is None else acc + q
                return acc / len(TABM_SEEDS)

            p_ord[rows_f] = 0.6 * r4_avg("all") + 0.4 * r4_avg("futures")
'''


def grab(src, name):
    lines = src.splitlines()
    s = next(i for i, ln in enumerate(lines) if ln.startswith(f"def {name}("))
    e = s + 1
    while e < len(lines) and not (lines[e] and not lines[e][0].isspace()):
        e += 1
    return "\n".join(lines[s:e]).rstrip() + "\n"


def main():
    z20 = zipfile.ZipFile(SRC20)
    src = z20.read("script.py").decode("utf-8")
    s19 = zipfile.ZipFile(SRC19).read("script.py").decode("utf-8")

    src = src.replace("CALIB_LOGIT_SHIFT = -0.004",
                      f"CALIB_LOGIT_SHIFT = {SHIFT}\n"
                      f"TABM_SEEDS = {tuple(SEEDS)}")
    assert "TABM_SEEDS" in src, "시프트 줄을 못 찾았다"

    # submit_20 의 퓨처스 블록을 통째로 교체한다. 1군 블록은 손대지 않는다.
    a = src.index("        rows_f = np.flatnonzero(isf_o)")
    b = src.index("        p_ord = np.clip(p_ord, 0.0, 1.0)")
    src = src[:a] + NEW_FUT + "\n" + src[b:]

    for fn in ("_logit", "_tabm_forward"):
        if f"def {fn}(" not in src:
            src = src.replace("\ndef load_test(",
                              "\n" + grab(s19, fn) + "\n\ndef load_test(", 1)

    assert "fbregime_" not in src, "구 퓨처스 모델 참조가 남았다"
    assert "CALIB_T" in src, "온도 상수가 사라졌다 (submit_20 은 T 를 쓴다)"

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "model"))
    io.open(os.path.join(OUT, "script.py"), "w", encoding="utf-8",
            newline="\n").write(src)
    # submit_20 에서 그대로 가져오는 것 — 퓨처스 모델만 빼고 전부
    keep = [n for n in z20.namelist()
            if n != "script.py" and "fbregime_" not in n]
    for n in keep:
        p = os.path.join(OUT, n)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        io.open(p, "wb").write(z20.read(n))
    for br in ("all", "futures"):
        for sd in SEEDS:
            f = os.path.join(DL, f"fin_{br}_reg4_s{sd}.npz")
            if not os.path.exists(f):
                raise SystemExit(f"없음: {f}")
            shutil.copy(f, os.path.join(OUT, "model", f"r4_{br}_s{sd}.npz"))

    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(OUT) for f in fs)
    print(f"  {OUT}  {tot/1e6:.1f}MB   시드 {SEEDS}   shift={SHIFT}")
    print(f"  submit_20 에서 그대로 가져온 파일 {len(keep)}개")
    for r, _, fs in os.walk(OUT):
        for f in sorted(fs):
            p = os.path.join(r, f)
            rel = os.path.relpath(p, OUT)
            tag = "  <- 새로 학습" if rel.startswith(("model\\r4_", "model/r4_")) \
                else ("  <- 새로 씀" if rel == "script.py" else "")
            print(f"    {rel:34s} {os.path.getsize(p)/1e6:7.2f}MB{tag}")


if __name__ == "__main__":
    main()
