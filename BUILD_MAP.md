# 1129 재구축 지도 — 산출물 ↔ 학습 스크립트

`submit_jaemin_54.zip` (LB 1129) 의 model/ 안 모든 파일을 어느 코드가 만드는지.
전체 재현 문서는 `handoff_1129/README.md` (필독), MS 기초는 `ms_base_1110.md`.
데이터는 대회 제공 train.csv 하나만 필요하다 (외부데이터 절대 금지).

## 산출물 → 생성 스크립트

| model/ 파일 | 생성 | 비고 |
|---|---|---|
| f2b_all/regular/futures_s42.npz | `colab/export_f2b_platfix_final.py` | 지인(train_chan_3) 전처리 경로 TabM, 3브랜치, 구퓨처스 0.1, S2(all 1에폭/퓨처스 4에폭). **시즌감쇠 없음이 사양** |
| msa_s42/s1/s777.npz | `colab/ms_deploy_audit.py` (env: `MSA_DECAY=1.5 MSA_S2=1 SEEDS=42,1,777`) → `colab/pt2npz.py` | MultiState (aux4). 배치는 시드 42,1 만 사용 |
| msa6_s42/s1.npz | 같은 스크립트 + `MSA_STATE6=1` | state6 판 (1군 라우팅용) |
| din_all/regular/futures_s{42,1,777}.npz + din_meta.npz | `colab/din_deploy.py` (`colab/ctr_zoo.py` 의 DIN) | 시즌감쇠 없음이 사양 (감쇠 실측 -6) |
| trees.npz, trees_base.npz (+ c12/cmh 표) | `nn_experiments/train_c12_submit.py` | CatBoost 50열, 시즌가중 2.0^, depth4/400iter, 트리를 numpy 배열로 export |
| hgb.npz | `colab/hgb_export.py` (+ 루트 `hgb_infer.py`) | HistGB 잔재 멤버 (비중 미미) |
| ms_plat_prior.npz, msa_cm48.npz | `colab/prior_plat_map.py` | MS 58열용 플래툰/카운트x월 동결 표 |
| history.json, config.json | `train_chan_3/` export 경로 산출물 | 지인 전처리의 as-of 복원 표 |
| warm_ids.npz | 한 줄: `np.savez_compressed("warm_ids.npz", pitcher_id=np.unique(train.pitcher_id).astype(np.int64))` | cold-투수 판정 |

## 추론/검증
| 파일 | 역할 |
|---|---|
| submit zip 안 `script.py` | numpy 추론 전체 (접힌 TabM GEMM, 트리 순회, 라우팅, 보정) |
| submit zip 안 `preprocess.py` | 지인 전처리의 추론 경로 (동결 표 lookup) |
| `colab/_v54.py` 등 `_v*.py` | 제출 전 검증 배터리 템플릿 (시간/규칙4/강제활성/zip) |
| `colab/features44.py`, `colab/multistate_softmax.py`, `colab/multistate_auditfeat.py`, `colab/tabm_gate_gpu.py` | 학습 공용 모듈 (44열 빌드 / MS 학습 / 감사판 피처 / TabM 학습기) |

## 학습 환경
- GPU + `torch 2.x`, `tabm`, `rtdl_num_embeddings`, `catboost`, `pandas`, `numpy`, `scipy`
- 실행 시 `PYTHONPATH` 에 repo 루트, `colab/`, `nn_experiments/` 추가, `AIMERS_DATA=<데이터경로>`
- 추론(제출물)은 numpy/pandas 만 (zip 의 requirements.txt)
- 시드 고정해도 GPU 가 다르면 바이트 재현은 안 됨. 점수 눈금 ±3 (시드운)

## 순서 (권장)
1. handoff_1129/README.md 정독 → A 경로(그대로 제출)로 1129 확인
2. 재학습은 한 계열씩: MS(§8 커맨드) → f2b → DIN → CB → 표
3. 매 변경 후 검증 배터리 (README §9) — 하나라도 건너뛰지 말 것
