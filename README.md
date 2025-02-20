# Keypoint Matching
## Install all libs
pip install -r requirements.txt

## Run Script
python -m torch.distributed.run --nproc_per_node=1 train_eval.py ./experiments/voc_basic.json
