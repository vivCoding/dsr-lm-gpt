# dsr-lm-gpt

a basic implementation of [paper](https://arxiv.org/abs/2305.03742) using chatGPT as model

### Setup
installing scallop as Python library
```
pip install scallopy-0.2.1-cp310-cp310-manylinux_2_31_x86_64.whl
```

### Pipeline
1. parse names
2. finetune gpt on relation extraction (sentence level)
3. test framework, gpt gives probabilities of each relationship `[1..20]`
3. pass into scallop, outputs relationship
4. simply result loss, no backpropagation
5. ???