# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/9 10:15
from torchvision.models import resnet50

net = resnet50()
for i in net.named_children():
    print(i)

