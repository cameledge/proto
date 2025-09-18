
# Building a Hydra-Driven Multimodal Prototyping Library with PyTorch Lightning and W&B

## Example Usage

- Train ResNet18 on FakeData:
`python scripts/train.py dataset=fake model=resnet18`

- Train ResNet18 on ImageNet:
`python scripts/train.py dataset=imagenet model=resnet18 dataset.root=/data/imagenet`

- Switch logger to CSV:b
`python scripts/train.py logger=csv`

- run with CPU-only CLIP and CPU LLM (gpt2 small)
`python scripts/multimodal_run.py model=multimodal model.clip.device=-1 model.llm.device=-1 input.image_path=example.jpg`

- If you have a GPU and want CLIP on GPU (cuda:0) and LLM on GPU:
`python scripts/multimodal_run.py model=multimodal model.clip.device=0 model.llm.device=0 input.image_path=/data/imgs/1.jpg`

- Train multimodal model (CLIP + LLM) on example image:
`python scripts/train.py model=multimodal input.image_path=example.jpg`

- To train over a batch of images
`python scripts/multimodal_batch.py model=multimodal input_batch.image_folder=./images`

optional overrides
```
python scripts/multimodal_batch.py model=multimodal \
    input_batch.image_folder=./my_imgs \
    input_batch.top_k=5 \
    input_batch.prompt_template="Labels: {labels}. Describe in detail for research analysis."
```

