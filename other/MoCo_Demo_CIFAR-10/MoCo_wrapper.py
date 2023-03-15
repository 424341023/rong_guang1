# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/2 17:06
import torch.nn as nn
import torch
from argss import args
from base_encoder import ModelBase
import torch.nn.functional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelMoCo(nn.Module):
    ''' 1）base_encoder，是查询编码器和关键字编码器所使用的网络结构；
        2) dim，通过编码器提取出来的特征的维度；
        3）K，字典队列的大小()，相当于存储了多少个128维的特征
        4）m，动量更新的参数；
        5）T，温度参数；'''
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=True):
        super(ModelMoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)

        # zip函数将两个可迭代的对象中的元素对应打包成元组，然后将这些元组以list的形式返回，长度取决于较短的对象
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        # 创建字典队列和队列指针，使用指针来完成新样本入队列，旧样本出队列。
        '''在这里创建队列和队列指针的时候使用了register_buffer方法，这个方法是干什么用的呢？
           在pytorch的参数包含两种，一个是nn.Parameter，给参数在loss.backward，optimizer.step之后进行更新；
           另外一个是buffer，通过register_buffer注册buffer，该参数在forward的时候进行更新。
           并且在模型的保存和加载时也会对于buffer进行写入和读出。
        '''
        self.register_buffer("queue", torch.randn(dim, K)) # 固定维度 [128, 4096]
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # 在最初的模型初始化时，encoder_q和encoder_k的参数是相同的，所以此时无论m为多少
            # encoder_k的参数不会发生变化
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # 判断字典的大小是否是batch_size的整数倍
        assert self.K % batch_size == 0  # for simplicity

        '''进队和入队则是根据指针来完成的，通过指针记录当前队列为空的索引位置，
           如果队列未满，则直接入队，如果队列已满，通过取余操作让新的batch覆盖掉旧的batch，然后更新指针的值。
        '''
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # 随机打乱索引值 random shuffle index
        # torch中没有random.shuffle
        # y = torch.randperm(n) y是把1到n这些数随机打乱得到的一个数字序列
        idx_shuffle = torch.randperm(x.shape[0]).to(device)

        # 存储未打乱时的索引值  index for restoring
        # 返回能使得数据有序的索引，比如torch.argsort([2, 1])，结果是[1, 0]
        idx_unshuffle = torch.argsort(idx_shuffle)

        # 返回打乱索引后的x的值，以及未打乱时的索引值
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # q:im_q=im_1, k:im_k=im_2
        # 计算q的特征  compute query features
        q = self.encoder_q(im_q)  # queries: NxC [512, 128]
        # 对q进行归一化，
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # 计算k的特征  compute key features
        with torch.no_grad():  # no gradient to keys
            # 打乱是为了使用BN shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC [512, 128]
            # 对k进行归一化
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        '''
        分别计算了q与正样本对k的乘积，然后计算了q与字典队列中的负样本的乘积，这里利用了爱因斯坦求和法，
        对于'nc,nc->n'是将每一行对应列的位置相乘求和得到n*1的向量，'nc,ck->nk'就是简单的矩阵乘法。'''
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        loss = nn.CrossEntropyLoss().to(device)(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # 加载完当前batch的数据后，先更新关键字编码器encoder_k
        # 该更新过程不需要进行梯度的计算
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # 模型的参数更新完毕后，开始计算损失
        # compute loss
        if self.symmetric:  # symmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss
