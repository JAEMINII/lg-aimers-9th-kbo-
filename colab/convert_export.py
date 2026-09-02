import sys
from pathlib import Path
sys.path.insert(0,'/root/aimers/train_chan_3')
from build_submission import export_one
out=Path('/root/ms_export')
for seed in (42,1,777):
    print(export_one(out/f'ms_seed{seed}.pt',out/f'ms_seed{seed}.npz'))
