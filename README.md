# NMT: Normalized Matching Transformer [(arxiv)]()
Normalized Matching Transformer (NMT) is an end-to-end deep learning pipeline that fuses a swin-transformer backbone, a SplineCNN for geometry-aware keypoint refinement, and a normalized transformer decoder with Sinkhorn matching and advanced contrastive/hyperspherical losses to achieve state-of-the-art sparse keypoint correspondence.

![NMT architecture during inference](./misc/NMT_inference.png)
![NMT architecture during training](./misc/NMT_train.png)
![Examples](./misc/examples.png)

## Results

### PascalVOC
![pascalVOC results](./misc/pascalVOC.png)

### Spair-71k
![spair results](./misc/spair.png)

## Requirements
We use `CUDA 12.4` and `GCC 11.4.0`. All needed packages and libraries are in `environment.yml`.

### Download datasets
Run the `download_data.sh` script.

### Backbone weights
We use the SwinV2 model as our backbone. You need to download the SwinV2-L* weights, which were pretrained on `ImageNet-22K` and finetuned on `ImageNet-1K` [(SwinV2)](https://github.com/microsoft/Swin-Transformer).
The weights location path should be `./utils/checkpoints/`.

## Installation
### conda env.
1. Entry the path of your conda environment folder in the last line of the `environment.yml` file.
2. Entry the command: 
```bash 
conda env create -f environment.yml
```
## Usage

### Parameters
All Parameters are in the `experiments/` folder.

### Running Training / Evaluation
```bash
python -m torch.distributed.run --nproc_per_node=1 train_eval.py ./experiments/voc_basic.json
```
- `--nproc_per_node=1` sets how many GPUs you want to run the model on (for now use 1 GPU to get same results).
- `./experiments/voc_basic.json` is which Parameters and Dataset to use. Other option would be `./experiments/spair.json`.
