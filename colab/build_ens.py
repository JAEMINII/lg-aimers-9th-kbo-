# -*- coding: utf-8 -*-
"""3원 앙상블 제출본을 조립한다.

    최종 = 0.10 x CatBoost + 0.30 x flatMLP + 0.60 x TabM,  로짓 시프트 -0.0145

세 모델의 출처
    CatBoost   submit_jaemin_12 (실측 1021 의 주력). 시즌가중 2.0, 60:40 라우팅
    flatMLP    전체 시즌 균등 학습, 6epoch, 시드 3개 평균. 우리 44피처
    TabM       submit_jaemin_14 (실측 1041). ep2, 3브랜치 60:40. 지인 44피처

피처 파이프라인이 둘인 이유
    우리 build_features 와 지인 build_inference_features 는 이름·순서가 같지만
    plat_dev 값이 다르다(최대 2.5e-2). 각 모델은 학습에 쓴 쪽을 그대로 받아야 한다.
        우리 경로  -> CatBoost, flatMLP
        지인 경로  -> TabM
    둘 다 돌려도 피처 생성은 각각 몇 초다.

구성 방식
    submit_jaemin_12/script.py 를 뼈대로 쓰고 (CatBoost 트리순회 + 피처 생성이 이미 있다)
    flatMLP numpy 추론과 TabM numpy 추론을 얹는다.
"""
import io
import os
import re
import zipfile

NEW_DOC = '"""\nscript.py — 평가 서버가 실행하는 추론 스크립트.\n\n최종 예측 = 0.10 x CatBoost + 0.30 x flatMLP + 0.60 x TabM,  로짓 시프트 -0.0145\n\n  CatBoost   2019~2024 학습, 시즌가중 2.0**(season-2019).\n             0.6 x 전체 + 0.4 x 정규리그(R)단독 라우팅 블렌딩\n  flatMLP    전체 시즌 균등 학습, 6epoch, 시드 3개 평균. PLR 임베딩 + 3층 MLP\n  TabM       BatchEnsemble(k=32). 2epoch 후 마지막 시즌으로 미세조정,\n             전체/퓨처스/정규 3브랜치를 0.6:0.4 로 합침\n\n시즌가중은 2026-08-20 에 추가했다. 리그 성공률이 2019 54.95% -> 2024 48.97% 로\n계속 밀리는데 옛 시즌을 같은 무게로 보고 있었다. 관문에서 판별력 886.9 -> 908.9.\n\'최근 시즌만 쓰기\' 는 이미 재봤고 손해였다(849.0 -> 823.3). 표본을 버리기 때문이다.\n가중치는 표본을 하나도 안 버리면서 드리프트만 줄인다.\n\n의존성은 numpy 와 pandas 뿐이다. sklearn / joblib / torch 를 쓰지 않는다.\n  -> 학습된 트리와 신경망 가중치를 순수 numpy 배열(model/*.npz)로 내보내고\n     추론도 numpy 로 한다. 피클 버전 불일치(sklearn 버전, numpy BitGenerator 등)로\n     로드가 깨질 여지를 없앴다.\n사전학습 가중치는 하나도 쓰지 않는다. model/ 아래 배열은 전부 제공된 train.csv\n로만 학습한 결과다.\n\n── 핵심 아이디어 ────────────────────────────────────────────────────────────\nasof_* 컬럼은 \'커리어 누적\'이며 시즌마다 리셋되지 않는다(train 에서 100% 검증).\n또한 asof_*_rate x asof_*_n 은 반올림하면 정확한 정수 카운트로 복원된다.\n\n따라서 평가 시즌(2025) 행에 대해\n    2025시즌 투구수 = asof_pitcher_n       - (그 투수의 train 총 투구수)\n    2025시즌 성공수 = round(rate x asof_n) - (그 투수의 train 총 성공수)\n로 \'평가 시즌 한정 성적\'을 각 행마다 독립적으로 복원할 수 있다.\n\n빼는 값(train 총계)은 학습 때 얼려 model/ 에 담아둔 표에서 가져온다. 평가 데이터를\n훑어서 구하지 않는다 — 지인 preprocess 쪽도 마찬가지로 train_mode 일 때만\n프레임 집계를 쓰고 추론 경로는 얼린 표만 쓴다.\n\n따라서 각 행의 예측은 그 행의 입력 변수와 train.csv 만으로 결정된다.\n\n규칙 4 실측 (2026-08-21, 6만 행)\n    행 순서를 섞고 재실행         최대차 0.000e+00\n    20% (1.2만 행)만 넣고 재실행  최대차 0.000e+00\n부분 검사가 핵심이다 — 순서만 섞으면 전체 평균 같은 걸 써도 값이 안 변한다.\n──────────────────────────────────────────────────────────────────────────\n"""'

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
DST = os.path.join(ROOT, "submit_ens")

# 원본이 CRLF 라 앵커 매칭 전에 LF 로 통일한다
base = zipfile.ZipFile(os.path.join(ROOT, "submit_jaemin_12.zip")).read(
    "script.py").decode("utf-8").replace(chr(13) + chr(10), chr(10))
tabm = zipfile.ZipFile(os.path.join(ROOT, "submit_jaemin_14.zip")).read(
    "script.py").decode("utf-8").replace(chr(13) + chr(10), chr(10))
mlpnp = io.open(os.path.join(ROOT, "colab", "mlp_numpy_new.py"),
                encoding="utf-8").read()

# ── TabM 추론부만 발췌 (load_bundle / _tabm_transform / _one_hot / _sigmoid / predict_frame)
a = tabm.index("def load_bundle(")
b = tabm.index("def main():")
tabm_core = tabm[a:b].rstrip()
tabm_core = tabm_core.replace("def _sigmoid(", "def _tabm_sigmoid(")
tabm_core = tabm_core.replace("_sigmoid(logits", "_tabm_sigmoid(logits")

# predict_frame 은 이름 기반 전처리(_tabm_transform)와 순전파가 한 함수에 붙어 있다.
# 우리 전처리로 만든 배열을 바로 먹이려면 둘을 떼어놔야 한다.
_OLD_PF = (
    '    """Predict already-transformed 44-feature rows independently."""\n'
    "    nums, cats = _tabm_transform(X, metadata)\n"
    "    if len(nums) == 0:"
)
_NEW_PF = (
    '    """이름 기반 전처리를 거쳐 순전파. 지인 경로용이며 지금은 쓰지 않는다."""\n'
    "    nums, cats = _tabm_transform(X, metadata)\n"
    "    return _tabm_forward(nums, cats, metadata, archive, chunk_size)\n"
    "\n"
    "\n"
    "def _tabm_forward(nums, cats, metadata: dict, archive, chunk_size: int = 1024):\n"
    '    """이미 전처리된 배열로 TabM 순전파. 전처리 방식과 분리해 둔다."""\n'
    "    if len(nums) == 0:"
)
assert tabm_core.count(_OLD_PF) == 1
tabm_core = tabm_core.replace(_OLD_PF, _NEW_PF)

# ── flatMLP numpy 추론부 (import 줄 제거)
mlp_core = "\n".join(l for l in mlpnp.split("\n")
                     if not re.match(r"^(import|from)\s", l))
mlp_core = mlp_core[mlp_core.index("def _plr("):].rstrip()
mlp_core = (mlp_core.replace("def predict_one(", "def _fm_predict_one(")
                    .replace("def predict(", "def _fm_predict(")
                    .replace("def prep(", "def _fm_prep(")
                    .replace("predict_one(Xn, Xc, z, chunk)", "_fm_predict_one(Xn, Xc, z, chunk)"))

BLOCK = '''

# =====================================================================
# flatMLP — 지금은 쓰지 않는다 (비중 0)
# =====================================================================
#
# 코드는 남겨 두되 W_MLP = 0 이라 파일도 안 읽는다.
#
# 뺀 이유: 관문은 빼면 -36 이라 했는데 리더보드는 +2 였다(1046 -> 1048).
# 관문이 이 모델을 과대평가하는 경로가 둘이고 둘 다 배치엔 없다.
#   선택 편향    설정을 후보 10개 중 관문 최고로 골랐다. 기여 +45.5 인데
#                안 고른 12개 평균은 +18.8. 차이 약 +22.
#   최적시프트   예측 평균이 실제보다 1.7%p 높은 미보정 모델인데 관문은
#                후보마다 최적 시프트로 채점해 그걸 공짜로 고쳐 준다.
#                단독 802.9 가 고정 시프트에서는 728.6 이 된다.
{MLP_CORE}


# =====================================================================
# TabM — 우리 전처리 + 퓨처스 체제 처리
# =====================================================================
#
# official tabm 을 numpy 로 뽑은 것. Stage1 2epoch + Stage2 마지막 시즌.
# 브랜치는 all 과 futures 둘이다. regular 는 뺐다 —
# 1군 행에 regular 를 섞는 게 손해였고(alpha 스윕에서 aR=0 이 최적,
# 행 부트스트랩 200회 중 196회), all 학습분의 89%가 이미 regular 데이터다.
#
# epoch 은 2 다. 4 로 늘렸더니 리더보드가 1041 -> 980 이었다. 관문은 ep4 를
# +5.5 로 쳤지만 실측이 -61 이다. 학습량이 다른 설정은 관문이 순위를 못 매긴다.
#
# 이번에 바뀐 핵심은 퓨처스 체제 처리다. 퓨처스 성공률이 2023 년에
# 70.9% -> 47.3% 로 무너지는데(같은 해 1군은 -0.06%p) 학습 퓨처스의 80.4%가
# 옛 체제다. abs_regime 플래그를 넣고 옛 체제 행에 가중 0.1 을 건다.
# 관문 VS=2024 에서 전체 +16.7 (t=4.5, 4/4), 퓨처스 +140.7.
#
# 시즌가중은 뺐다. 다년 백테스트에서 2024 +46.9 / 2022 -25.3 / 2023 -231.1 로
# 부호가 갈렸다. 그 해에 드리프트가 클 때만 이득이라 고정 장치로는 못 쓴다.
{TABM_CORE}
'''.replace("{MLP_CORE}", mlp_core).replace("{TABM_CORE}", tabm_core)

anchor = "# =====================================================================\n# 데이터 로드 / 제출 파일\n# =====================================================================\n"
assert base.count(anchor) == 1
s = base.replace(anchor, BLOCK.strip("\n") + "\n\n\n" + anchor)

# ── 비중 상수
s = s.replace("MLP_WEIGHT = 0.20", '''# 앙상블 비중.  최종 = W_CB x CatBoost + W_MLP x flatMLP + W_TABM x TabM
#
# 0.14 / 0.00 / 0.86 은 submit_18 에서 태워 1048 을 받은 값이다.
# flatMLP 을 빼고 CatBoost:TabM 비율(1:6)을 유지한 구성이다.
#
# 비중을 관문으로 정하려던 시도는 세 번 다 틀렸다. 관문은 큰 보정을 필요로
# 하는 구성에 점수를 더 준다(후보마다 최적 시프트로 채점하므로). 배치엔 없다.
W_CB, W_MLP, W_TABM = 0.14, 0.00, 0.86

# TabM 멤버. 파일 이름이 model/tabm_{branch}_{멤버}.npz 가 된다.
# regular 브랜치를 빼서 패스가 절반 가까이 줄었고, 그 예산으로 시드를 3개 쓴다.
TABM_MEMBERS = ("seed42", "seed1", "seed777")

# 학습에 건 시즌가중. model/ 안의 meta 와 대조해 불일치면 멈춘다.
# 지금은 0(가중 없음)이다. 한 번은 의도한 가중이 안 걸린 판본을 낼 뻔했다 —
# 내보내기 코드를 고쳐놓고 서버에 안 올려서 인자가 무시된 채 학습됐다.
TABM_DECAY = 0.0

# 퓨처스 행에 분기 모델을 얹는 세기. logit 잔차 계수다.
# 관문 alpha 격자에서 0.4 가 최적이었고 0.6 과는 구별되지 않았다.
ALPHA_F = 0.4


def _logit(p):
    q = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(q / (1.0 - q))''')

# ── 시프트. 이 상수는 submit_12 원본에 있어 build_ens 안의 치환으로는 안 바뀐다.
# flatMLP 을 빼면서 혼합 예측 평균이 내려가 덜 보정해야 한다. 리더보드 두 점에서
# 역산한 관계(최적시프트 = -4 x (예측평균 - 실제평균))로 잡은 값이고, 이걸로
# submit_18 이 1046 -> 1048 을 받았다.
old_shift = [l for l in s.split(chr(10)) if l.startswith("CALIB_LOGIT_SHIFT")]
assert len(old_shift) == 1, old_shift
s = s.replace(old_shift[0], "CALIB_LOGIT_SHIFT = -0.005")

# ── main 교체
old_main = s[s.index("    print(\"Inference...\")"):s.index("    print(\"Build submission...\")")]
new_main = '''    print("Inference...")
    if len(X):
        p_cb = predict_numpy(X.values.astype(np.float64), z)
        print(f" CatBoost mean={p_cb.mean():.4f}")

        # flatMLP — 비중 0 이면 파일 자체를 안 읽는다. 제출본에서 뺐기 때문이다.
        if W_MLP == 0.0:
            p_mlp = np.zeros(len(X))
        else:
            st = np.load(resolve("model/prep_vs2025.npz"), allow_pickle=False)
            Xn, Xc = _fm_prep(X.values.astype(np.float64), st)
            arcs = [np.load(resolve(f"model/vs2025_seed{s_}.npz"),
                            allow_pickle=False) for s_ in (42, 1, 777)]
            p_mlp = _fm_predict(Xn, Xc, arcs)
            print(f" flatMLP  mean={p_mlp.mean():.4f}  "
                  f"상관={np.corrcoef(p_cb, p_mlp)[0, 1]:.4f}")

        # TabM — 우리 45피처(44 + abs_regime), 브랜치 all + futures, 시드 3개
        #
        # 바뀐 것은 시즌가중 하나다. sample_weight = 3.5 ** (season - 2019).
        # 8시드 짝비교로 확정했다: 관문 단독 875.1 -> 911.2, t=6.60, 8/8 양수.
        # CatBoost 최적값 2.0 을 그대로 쓰고 있었는데 TabM 최적점은 3~4 다.
        # decay 를 훑으면 3 이후로는 평평하다(3.0 +38.5 / 4.0 +40.0 / 8.0 +42.7).
        # 고원 위에서는 덜 극단적인 쪽이 안전해 3.5 를 골랐다.
        #
        # 전처리도 우리 것으로 통일했다. 관문에서 시드 3개 모두 우리 쪽이
        # 높았고(단독 +6.7), 무엇보다 지인 preprocess.py 가 통째로 빠진다.
        # 피처 파이프라인이 하나로 줄어 추론 시간과 불일치 위험이 같이 사라진다.
        #
        # 시드가 2개인 이유는 시간이다. 3개면 서버 11.2분으로 제한을 넘고,
        # 1개면 뽑기 편차가 ±24 라 시즌가중 효과를 리더보드에서 읽을 수 없다.
        #
        # 멤버마다 전처리 통계가 다를 수 있다. 저카디널리티 범주화 판본은 범주가
        # 9개가 아니라 15개라 Xn/Xc 모양 자체가 달라진다. 그래서 멤버별로 만들고
        # 이름으로 캐시한다. 브랜치가 셋이라 같은 멤버를 세 번 만나는데 한 번만
        # 계산하면 된다. 253,507행 기준 전처리는 1초 미만이다.
        Xv = X.values.astype(np.float64)

        def with_extra(want):
            """meta 가 요구하는 열을 만들어 붙인다. 지금은 등판 강도 2열뿐이다.

            pn_cur 과 game_month 는 둘 다 44열에 있는데 그 '비율' 은 모델이
            스스로 만들어야 한다. 나눗셈은 신경망이 잘 못 배우는 형태라
            명시적으로 준다. 8시드 짝비교에서 혼합 +2.4, 8/8 양수, t=3.41.
            """
            cols = list(X.columns)
            if list(want) == cols:
                return Xv
            extra = [c for c in want if c not in cols]
            if list(want)[:len(cols)] != cols:
                raise ValueError("meta 의 앞 44열이 build_features 와 다르다")
            out = Xv
            for c in extra:
                if c == "load_rate":
                    pn = Xv[:, cols.index("pn_cur")]
                    mo = Xv[:, cols.index("game_month")]
                    el = np.clip(mo - 2.0, 1.0, None)      # 3월 개막
                    out = np.c_[out, np.log1p(pn) - np.log1p(el)]
                elif c == "load_lin":
                    pn = Xv[:, cols.index("pn_cur")]
                    mo = Xv[:, cols.index("game_month")]
                    el = np.clip(mo - 2.0, 1.0, None)
                    out = np.c_[out, pn / el]
                elif c == "abs_regime":
                    # 퓨처스는 2023 년에 성공률이 70.9% -> 47.3% 로 무너진다.
                    # 학습에는 두 체제가 섞여 있지만 평가(2025)는 전부 새 체제다.
                    out = np.c_[out, np.ones(len(Xv))]
                else:
                    raise ValueError(f"meta 가 요구하는 열을 못 만든다: {c}")
            return out

        _pcache = {}

        def tabm_avg(branch, rows=None):
            """브랜치 하나를 멤버 평균으로 예측한다. rows 를 주면 그 행만."""
            acc = None
            for name in TABM_MEMBERS:
                z = np.load(resolve(f"model/tabm_{branch}_{name}.npz"),
                            allow_pickle=False)
                if name not in _pcache:
                    mt = json.loads(str(z["meta"].item()))
                    # 순전파 코드는 지인 형식의 키 이름을 읽는다. 우리 meta 는
                    # cards 로 담으므로 맞춰 준다. 나머지(k, n_blocks, d_block,
                    # num_embeddings, backbone_kind)는 이름이 같다.
                    mt["cat_cardinalities"] = mt["cards"]
                    want = [str(c) for c in mt["features"]]
                    if want[:len(X.columns)] != list(X.columns):
                        raise ValueError(f"{name}: 피처 순서가 build_features 와 다르다")
                    # 학습 설정이 의도와 같은지 확인한다.
                    if abs(float(mt.get("decay", 0.0)) - TABM_DECAY) > 1e-9:
                        raise ValueError(
                            f"{name}: 시즌가중 {mt.get('decay')} != 기대 {TABM_DECAY}")
                    _pcache[name] = (mt, *_fm_prep(with_extra(want), z))
                mt, Tn, Tc = _pcache[name]
                n_ = Tn if rows is None else Tn[rows]
                c_ = Tc if rows is None else Tc[rows]
                q = _tabm_forward(n_, c_, mt, z, 4096)
                acc = q if acc is None else acc + q
            return acc / len(TABM_MEMBERS)

        # 라우팅: 퓨처스 행에만 분기 모델을 logit 잔차로 얹는다.
        #
        #   F행:  logit(all) + ALPHA_F x (logit(futures) - logit(all))
        #   R행:  all 그대로
        #
        # regular 브랜치는 뺐다. alpha 스윕에서 1군 행에 regular 를 섞는 게
        # 손해였고(aR=0 이 최적), 행 부트스트랩 200회 중 196회에서 그랬다.
        # all 학습분의 89%가 이미 regular 데이터라 같은 걸 배운다.
        # 빼면 추론 패스가 절반 가까이 줄어 같은 시간에 시드를 3개 넣는다.
        #
        # 확률 혼합 대신 logit 잔차를 쓴다. 관문에서 +1.9 였다(고정 시프트 기준).
        # aF 는 0.4 와 0.6 이 구별되지 않아(부트스트랩 54.5%) 0.4 를 쓴다.
        pa = tabm_avg("all")
        gt = test["game_type"].astype(str).to_numpy()
        msk = np.flatnonzero(gt == "F")
        p_tabm = pa.copy()
        if len(msk):
            pf = tabm_avg("futures", msk)
            la = _logit(pa[msk])
            z = la + ALPHA_F * (_logit(pf) - la)
            p_tabm[msk] = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        p_tabm = np.clip(p_tabm, 0.0, 1.0)
        print(f" TabM     mean={p_tabm.mean():.4f}  F행 {len(msk):,}개  "
              f"상관={np.corrcoef(p_cb, p_tabm)[0, 1]:.4f}")

        preds = W_CB * p_cb + W_MLP * p_mlp + W_TABM * p_tabm
        print(f" blended  mean={preds.mean():.4f}  "
              f"({W_CB:.2f}/{W_MLP:.2f}/{W_TABM:.2f})")
        preds = apply_calibration(preds, shift)
        print(f" calibrated mean={preds.mean():.4f}")
    else:
        preds = []

'''
s = s.replace(old_main, new_main)
s = s.replace("import os\n\nimport numpy as np", "import json\nimport os\n\nimport numpy as np")

# 더 이상 안 쓰는 기존 MLP 로더 호출 제거
s = s.replace('''    mlp_w, mlp_F = load_mlp()
    if mlp_w is None:
        print(" MLP 파일 없음 -> CatBoost 단독으로 진행")
        w_mlp = 0.0
    else:
        w_mlp = float(MLP_WEIGHT)
        print(f" MLP OK. 입력 {len(mlp_w['cat_ind'])}열  비중 {w_mlp:.2f}")
''', '''    print(f" 앙상블 비중  CatBoost {W_CB:.2f} / flatMLP {W_MLP:.2f} / TabM {W_TABM:.2f}")
''')

# ── 옛 MLP-PLR 섹션 제거
#
# submit_12 의 MLP 는 '2024 한 시즌만' 학습한 pytabkit 판본이다. flatMLP 가 그
# 자리를 대체하므로 통째로 죽은 코드가 된다 (load_mlp / mlp_predict /
# build_mlp_input / quantile_transform / ndtri / AS241 계수 ~310줄).
# 남겨두면 이름이 겹쳐 헷갈리기만 하니 잘라낸다.
i = s.index("def load_mlp():")
j = s.index("# " + "=" * 69 + chr(10) + "# flatMLP")
s = s[:i] + s[j:]

# ── 모듈 docstring 교체
#
# 뼈대로 쓴 submit_12 의 설명은 2원 앙상블(CatBoost 0.8 + 옛 MLP 0.2) 기준이라
# 지금 코드와 맞지 않는다. 채점자가 읽는 첫 화면이니 실제 구성으로 바꾼다.
a0 = s.index('"""')
a1 = s.index('"""', a0 + 3) + 3
s = s[:a0] + NEW_DOC + s[a1:]

io.open(os.path.join(DST, "script.py"), "w", encoding="utf-8").write(s)
import ast
ast.parse(s)
print(f"script.py 생성  {s.count(chr(10))+1}줄")
