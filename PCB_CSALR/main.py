from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.trainers_partloss_4stage import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

'''
This is the code for paper 'parameter-free spatial attention network for Person Re-Identification'
Our code is mainly based on PCB 
'''
def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(dataset.train, root=osp.join(dataset.images_dir,dataset.train_path),
                    transform=train_transformer,random_mask=True),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query, root=osp.join(dataset.images_dir,dataset.query_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir,dataset.gallery_path),
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)


    return dataset, num_classes, train_loader, query_loader, gallery_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset,  args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 )


    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)

    model = nn.DataParallel(model).cuda()

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        # model_dict = model.state_dict()
        # checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        # model_dict.update(checkpoint_load)
        # model.load_state_dict(model_dict)
        model.module.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}".format(start_epoch, best_top1))

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        print(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        model.module.load_state_dict(checkpoint['state_dict'])
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    if hasattr(model.module, 'base'):
        # base_param_ids = set(map(id, model.module.base.parameters()))
        # new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
        # param_groups = [
        #     {'params': model.module.base.parameters(), 'lr_mult': 0.1},
        #     {'params': new_params, 'lr_mult': 1.0}]
        base_param_ids = set(map(id, model.module.base.parameters()))
        attn_params    = list(model.module.SA0.parameters()) + list(model.module.SA1.parameters()) + list(model.module.SA2.parameters()) + list(model.module.SA3.parameters()) + list(model.module.SA4.parameters())
        attn_param_ids = set(map(id, attn_params))
        # new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
        new_params   = [p for p in model.parameters() if (id(p) not in base_param_ids) and (id(p) not in attn_param_ids)]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': attn_params, 'lr_mult': 0.1},
            {'params': new_params,  'lr_mult': 1.0},
        ]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # optimizer = torch.optim.Adam(param_groups,lr=args.lr)
    # Trainer
    trainer = Trainer(model, criterion, 0, 0, SMLoss_mode=0)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else args.step_size
        lr = args.lr * (0.1 ** (epoch // step_size))
        # if epoch>70:
        #     lr = 0.01
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        is_best = True
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=384,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--step-size',type=int, default=40)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())