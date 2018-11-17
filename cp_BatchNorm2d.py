import torch
from torch.nn import functional as F


class CpBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(CpBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        self._check_input_dim(input)
        if input.requires_grad:
            exponential_average_factor = 0.0
            if self.training and self.track_running_stats:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / self.num_batches_tracked.item()
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, 0.0, self.eps)
