# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/2 20:57
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from argss import args


# 数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

""" Training dataset"""
class TrainPair(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # self.len = train.shape[0]
        # self.data = train
        # self.classes = np.max(train_label)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.uint8(img))

        if self.transform != None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)
        return im_1, im_2

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" memory dataset"""
class TrainDS(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # self.len = train.shape[0]
        # self.data = train
        # self.targets = train_label
        # self.classes = np.max(train_label)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)

        target = self.targets[index]
        # 根据索引返回数据和对应的标签
        return img, target

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Test dataset"""
class TestDS(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # self.len = test.shape[0]
        # self.data = test
        # self.targets = test_label
        # self.classes = np.max(train_label)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)
        target = self.targets[index]

        # 根据索引返回数据和对应的标签
        return img, target

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# # 创建 trainloader 和 testloader
train_data = TrainPair(transform=train_transform)
memory_data = TrainDS(transform=test_transform)
test_data = TestDS(transform=test_transform)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                           pin_memory=True, drop_last=True)
memory_loader = DataLoader(dataset=memory_data, batch_size=args.batch_size, shuffle=False,
                                            num_workers=0, pin_memory=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                          pin_memory=True, drop_last=True)
