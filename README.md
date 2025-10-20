# ViT-GPU-Project
GPU computing course project 2025

## How to
### Prerequisites
```bash
python -m venv gpu-open-vit
source gpu-open-vit/bin/activate
pip install -r requirements.txt
```

### Compile

```bash
make test_bin/<file>.exe
```

### Create the dataset

### Train the models

### Launch the runs
*Local*:
To launch the test and generate the json `bash sect_bench.sh` *deprecated*

*Cluster*:
```bash
bash sbatchman_block_test.sh <block_name>
```
### Generate the json files


### Plot the results
```bash
python3 scripts/
```
## Objective

*"The goal of this project is to accelerate key components of the Open-ViT-Bench Visual Transformer models using CUDA. The focus will be on parallelizing the self-attention mechanism, patch embedding, and matrix multiplications involved in the transformer blocks. Feel free to leverage optimized libraries for tasks like GEMM."*

1. Matrix multiplication where presents
   1. f
   2. f
   3. f
2. Patch embedding
3. Self-attention

## Source material
- https://github.com/HicrestLaboratory/Open-VIT-bench (base repository)
- https://arxiv.org/pdf/2010.11929 (TRANSFORMERS FOR IMAGE RECOGNITION) 
- https://arxiv.org/pdf/1706.03762 (Attention is all you need)
- https://github.com/HicrestLaboratory/Open-VIT_hackathon/blob/master/cvit_structure.pptx (CViT slides)
- https://github.com/HicrestLaboratory/Open-VIT-bench/blob/main/vit-backbone.pptx (ViT backbone)


## Patch Embedding

It's a convolutional embedder, where for each input image, for each patch and do a convolution.
So compress each image patch in a embedding(token) of size 768.

16x16x3 => 1x768

## Self-Attention


