# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/2 16:53
# gpu_info = !nvidia-smi -i 0
# gpu_info = '\n'.join(gpu_info)
# print(gpu_info)

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from argss import args

'''
这里有三点需要注意：
    第一这里CIFAR10Pair 就是构造一个样本对。输出的target就是输入。所谓对比学习，无监督学习的精髓。
    第二这里的train_loader、memory_loader和test_loader。train和memory都是训练集的数据，
            它们的不同之处在于，数据增广的方式不同，数据的组成也不同。 
            train的增广是用来训练的，标签就是图像本身，
            memory是用来测试的，它和test是一致的，标签就是标注的标签。
            memory主要用于在测试的时候，如果用到 knn，则可以构造memory bank监控学习精度，
            test在测试的时候评估模型精度。
    第三这里可以改成自己的数据集。照猫画虎我的数据集：
    
'''
class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            # 因为每次变换都是随机的
            # 所以im_1和im_2是对一个批次中的每一个图像经过两种不同数据增强之后得到的图像
            # im_1和im_2是一正样本对(positive pair)
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# data prepare
train_data = CIFAR10Pair(root='data', train=True, transform=train_transform, download=True)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

memory_data = CIFAR10(root='data', train=True, transform=test_transform, download=True)
memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
