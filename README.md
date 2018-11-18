# pytorch-checkpoint
[Gradient checkpointing](https://github.com/openai/gradient-checkpointing) is a technique to reduce GPU memory cost.

## Official implementation
There exists a [PyTorch implementaion in the official repo](https://pytorch.org/docs/master/checkpoint.html).
However, it is extremely slow with multiple GPUs.

## This implementation
This repo contains a PyTorch implemention that can work on multiple GPUs.

## Main results

| Method | # GPU | Batch | Memory | Time  |
|--------|:----:|:-----:|:------:|:-----:|
|Naive|2|256| 5.25G   | 0.27s |
|Official|2|256|2.98G|1.41s|
|This repo|2|256|2.97G|0.31s|

## Documentation

```python
checkpoint.CheckpointFunction.apply(function, n, *args)
```

Parameters:	

  * function – describes what to run in the forward pass of the model or part of the model. It should also know how to handle the inputs passed as the tuple. For example, in LSTM, if user passes (activation, hidden), function should correctly use the first input as activation and the second input as hidden.
  * n – number of inputs to the function
  * args – tuple containing inputs to the function AND parameters to optimize in the function. Note that the first n elements in this tuple should be ordered inputs to the function. Other elements are considered as parameters.

Returns:	
  * Output of running function on inputs to the function
  
## Example
```python
# bn_function is a function containing conv1, norm1, relu1.
# naive no checkpointing: bottleneck_output = bn_function(*prev_features)
# official implementation: bottleneck_output = cp.checkpoint(bn_function, *prev_features)
args = prev_features + tuple(self.norm1.parameters()) + tuple(self.conv1.parameters())
# The parameters to optimize in the bn_function are tuple(self.norm1.parameters()) + tuple(self.conv1.parameters())
bottleneck_output = cp.CheckpointFunction.apply(bn_function, len(prev_features), *args)
```
Note: We recommend using checkpointing with cp_BatchNorm2d instead of torch.nn.BatchNorm2d, to avoid accumulating the same batch norm statistics more than once.

## Demo
[python-fire](https://github.com/google/python-fire) is not required for checkpointing, but is required for the efficient densenet demo.
```
pip install fire
```
To run the demo:
```
CUDA_VISIBLE_DEVICES=0,1 python cp_demo.py --efficient True --data cifar --save model --batch_size 256
```

## Environment
This code is tested with PyTorch 1.0.0.dev20181102

## Full results

| Method | # GPU | Batch | Memory | Time  |
|--------|:----:|:-----:|:------:|:-----:|
|Naive|1|256| 9.93G   | 0.42s |
|Naive|2|4| 0.65G   | 0.10s |
|Naive|2|256| 5.25G   | 0.27s |
|Naive|2|512| 9.93G   | 0.50s |
|Official|1|256|5.38G|0.52s|
|Official|1|512|10.1G|1.00s|
|Official|2|4|0.62G|1.40s|
|Official|2|256|2.98G|1.41s|
|Official|2|512|5.39G|1.53s|
|This repo|1|256|5.37G|0.50s|
|This repo|1|512|10.1G|0.97s|
|This repo|2|4|0.62G|0.13s|
|This repo|2|256|2.97G|0.31s|
|This repo|2|512|5.37G|0.58s|

## Credits

Part of our code in checkpoint.py and cp_BatchNorm2d.py is from https://github.com/pytorch/pytorch

The efficient densenet demo is taken from https://github.com/gpleiss/efficient_densenet_pytorch
