# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/2 17:15
# define optimizer
import torch
import os
import json
import pandas as pd
from model_train import train
from model_test import test
from data_loaders import train_loader, memory_loader, test_loader
from MoCo_wrapper import ModelMoCo
from argss import args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model
model = ModelMoCo(
    dim=args.moco_dim,
    K=args.moco_k,
    m=args.moco_m,
    T=args.moco_t,
    arch=args.arch,
    bn_splits=args.bn_splits,
    symmetric=args.symmetric,
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

# load model if resume
epoch_start = 1
if args.resume != '':
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))

# logging
results = {'train_loss': [], 'test_acc@1': []}
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)

# training loop
for epoch in range(epoch_start, args.epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch, args)
    results['train_loss'].append(train_loss)
    test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
    results['test_acc@1'].append(test_acc_1)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
    # save model
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), }, args.results_dir + '/model_last.pth')

