# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/2 16:59
import torch.nn.functional
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


class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        # 设置扩增批分割，如果扩增批分割(augmentation batch splits) 的数量大于1，
        # 将所有模型的BatchNormlayers转换为Split batch Normalization层。                                                        通常，当我们训练一个模型时，我们将数据增强【data augmentation】应用于完整批【batch】，然后从这个完整批定义批规范【batch norm】统计数据，如均值和方差。                      但有时将数据分成组，并为每个组使用单独的批处理归一化层【Batch Normalization layers】来独立地归一化组是有益的。这在一些论文中被称为辅助批规范【auxiliary batch norm】
        # 通常，当我们训练一个模型时，我们将数据增强(data augmentation)应用于完整批 (batch),
        # 然后从这个完整批(batchnorm)定义批规范统计数据，如均值(running_mean)和方差(running_var)。
        # 但有时将数据分成组，并为每个组使用单独的批处理归一化层 (Batch Normalization layers) 来独立地归一化组是有益的。
        # 这在一些论文中被称为辅助批规范(auxiliary batchnorm)
        # 例如，batch splits数量是2。那么数据拆分成两个组，一组没有数据增强，一组有数据增强。
        # 然后我们使用两个独立的批处理归一化层【Batch Normalization layers】来归一化两个组别
        # 在argss.py中默认设置的 bn_splits 为 8
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        # 在argss.py中默认设置的 arch 为 resnet18
        # 是个function，<function resnet18 at 0x000001AD560CC1E0>
        resnet_arch = getattr(resnet, arch)
        # 定义resnet_arch 中的类别数目，以及要采用的归一化层（相当于进行了实例化，有层有参数的实实在在的网络）
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            # replaces conv1 with kernel=3, str=1
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # removes pool1
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x
