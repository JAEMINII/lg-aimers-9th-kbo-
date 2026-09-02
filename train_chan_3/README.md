# train_chan_3

기존 `submit_chan_2` 전처리와 공식 TabM 설정을 유지하면서 다음 구조를 학습합니다.

- `all`: 전체 train
- `futures`: `game_type == F`
- `regular`: `game_type == R`
- 각 branch를 2024년 데이터로 마지막 MLP block + output layer fine-tuning
- 제출 시 `전체 0.6 + 해당 경기 유형 전용 0.4`

## Colab 실행

`train_conditional_colab.ipynb`를 Google Drive의 `aimers/train_chan_3`에 업로드하고
`PROJECT_DIR`만 실제 경로에 맞추면 됩니다. 마지막 셀에서 다음 ZIP을 생성합니다.

```text
aimers/submit_tabm_conditional.zip
```

제출 ZIP은 `script.py`, `preprocess.py`, `requirements.txt`, `model/*.npz`로 구성되며,
`script.py`는 각 행의 `game_type`에 따라 모델을 라우팅합니다.
