# pytorch-checkpoint
[Gradient checkpointing](https://github.com/openai/gradient-checkpointing) is a technique to reduce GPU memory cost.

## Previous implementation
There exists a [PyTorch implementaion in the official repo](https://pytorch.org/docs/master/checkpoint.html).
However, it is extremely slow with multiple GPUs.

## This implementation
This repo contains a PyTorch implemention that can work on multiple GPUs.

## Experiments

| Method | # GPU | Batch | Memory | Time  |
|--------|:----:|:-----:|:------:|:-----:|
|Naive|1|256| 9.93G   | 0.42s |
|Naive|2|4| 0.65G   | 0.10s |
|Naive|2|256| 5.25G   | 0.27s |
|Naive|2|512| 9.93G   | 0.50s |
|Previous|1|256|5.38G|0.52s|
|Previous|1|512|10.1G|1.00s|
|Previous|2|4|0.62G|1.40s|
|Previous|2|256|2.98G|1.41s|
|Previous|2|512|5.39G|1.53s|
|This|1|256|5.37G|0.50s|
|This|1|512|10.1G|0.97s|
|This|2|4|0.62G|0.13s|
|This|2|256|2.97G|0.31s|
|This|2|512|5.37G|0.58s|
