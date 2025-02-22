# Keypoint Matching
## Install all libraries
pip install -r requirements.txt

## Run Script
python -m torch.distributed.run --nproc_per_node=1 train_eval.py ./experiments/voc_basic.json
- nproc_per_node sets on how many GPU's to run
- voc_basic.json includes hyperparameters for PascalVOC training. See more in /experiments
