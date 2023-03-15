# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/2 21:47
# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/2 16:59
from functools import partial
from torch import nn
from torchvision.models import resnet


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


norm_layer = partial(SplitBatchNorm, num_splits=8)
# 在argss.py中默认设置的 arch 为 resnet18
resnet_arch = getattr(resnet, 'resnet18')
net = resnet_arch(num_classes=512, norm_layer=norm_layer)
for name, module in net.named_children():
    print(name, module)
print(resnet_arch)
# print(net)