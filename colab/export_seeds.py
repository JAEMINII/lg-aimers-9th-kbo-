# -*- coding: utf-8 -*-
"""지인 학습 코드로 만든 8시드 체크포인트를 npz 로 내보낸다.

왜 지인 코드로 학습하나
    리더보드가 학습 코드로 깨끗하게 갈린다.
        지인 1군   submit_15 1046  submit_18 1048  submit_20 1057  submit_27 1058
        우리 1군   submit_19 1020  submit_23 1048  submit_24 1049  submit_25 1052
    스케줄러 결함을 찾아 +21.6 을 회수하고도 4~5점이 남아 있다. 아직 못 찾았다.

    시드 곡선 실측이 k=1 -> k=4 에서 +8.0 이었는데, 우리 코드로 8시드를 만들면
    기반이 5점 낮은 데서 시작한다. 지인 코드를 그대로 여러 시드로 돌리면
    기반을 유지한 채 시드 이득만 얹을 수 있다. train_conditional.py 가
    --seeds 를 지원하므로 재구현 없이 된다.

바꾸는 것은 시드 개수 하나뿐
    submit_27(LB 1058)의 모델은 지인 코드 seed 42 하나다. 여기서 시드만
    8개로 늘린다. Stage2 에폭도 1 그대로 둔다 (우리 실험에서 퓨처스 4에폭이
    좋았지만 그건 우리 코드 얘기고, 여기서 같이 바꾸면 뭐가 효과인지 못 가린다).

내보내기
    build_submission.export_one 이 체크포인트 하나를 npz 로 바꾼다.
    브랜치 3개 x 시드 8개 = 24개를 만든다. 하나가 3.4MB 라 82MB,
    제출 예산 10GB 의 0.8% 다.
"""
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/aimers")
sys.path.insert(0, str(ROOT))

from train_chan_3.build_submission import export_one                # noqa: E402

ART = ROOT / os.environ.get("ART", "art8")
OUT = ROOT / os.environ.get("NPZ", "npz8")
SEEDS = ([42, 1, 777, 2] if os.environ.get("SEEDS4") else [42, 1, 777, 2, 7, 13, 99, 2024])
BRANCHES = ("all", "regular", "futures")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    made, missing = [], []
    for br in BRANCHES:
        for sd in SEEDS:
            ck = ART / f"{br}_stage2" / f"tabm_seed_{sd}.pt"
            if not ck.exists():
                ck = ART / f"{br}_stage1" / f"tabm_seed_{sd}.pt"
            if not ck.exists():
                missing.append(str(ck))
                continue
            dst = OUT / f"{br}_s{sd}.npz"
            export_one(ck, dst)
            made.append((f"{br}_s{sd}", dst.stat().st_size / 1e6))
    print(f"\n  내보낸 파일 {len(made)}개")
    for n, mb in made:
        print(f"    {n:16s} {mb:6.2f}MB")
    tot = sum(m for _, m in made)
    print(f"  합계 {tot:.1f}MB   / 10,240MB 예산 = {tot/10240*100:.2f}%")
    if missing:
        print(f"\n  없는 체크포인트 {len(missing)}개:")
        for m in missing[:10]:
            print("   ", m)
