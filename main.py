"""
CGCNN Training Script

Main training script for Crystal Graph Convolutional Neural Networks.
Supports regression, classification, and multi-task learning with
comprehensive evaluation and model checkpointing.

Author: lunazhang
Date: 2023
Reference: Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018)
"""

from __future__ import print_function, division

import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet, CrystalGraphConvNetMulti

# 设置随机种子
def set_random_seed(seed=123):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
    parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                        help='数据集路径和其他选项')
    parser.add_argument('--task', choices=['regression', 'classification', 'multi'],
                        default='regression', help='任务类型 (默认: regression)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='禁用CUDA')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='数据加载工作进程数 (默认: 0)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='手动设置开始轮数')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='batch size (默认: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='初始学习率 (默认: 0.01)')
    parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                        metavar='N', help='学习率调整里程碑 (默认: [100])')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='动量 (默认: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='权重衰减 (默认: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='打印频率 (默认: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='预训练模型路径')
    parser.add_argument('--cls-weight', default=1.0, type=float,
                        help='分类损失权重 (多任务学习时使用)')
    parser.add_argument('--train-ratio', default=0.6, type=float,
                        help='训练集比例 (默认: 0.6)')
    parser.add_argument('--val-ratio', default=0.2, type=float,
                        help='验证集比例 (默认: 0.2)')
    parser.add_argument('--test-ratio', default=0.2, type=float,
                        help='测试集比例 (默认: 0.2)')
    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                        help='选择优化器 SGD 或 Adam (默认: SGD)')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                        help='原子特征长度')
    parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                        help='隐藏特征长度')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                        help='卷积层数')
    parser.add_argument('--n-h', default=1, type=int, metavar='N',
                        help='全连接层数')

    args = parser.parse_args()

    # 设置随机种子
    set_random_seed()

    # CUDA设置
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    if args.task == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.

    # 加载数据
    dataset = CIFData(args.data_options[0])
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        return_test=True)

    # 获取数据特征
    sample_data_list = [dataset[i] for i in range(len(dataset))]
    _, sample_target, _ = collate_fn(sample_data_list)
    orig_atom_fea_len = sample_data_list[0][0].shape[-1]
    nbr_fea_len = sample_data_list[0][1].shape[-1]

    # 构建模型
    if args.task == 'multi':
        model = CrystalGraphConvNetMulti(orig_atom_fea_len, nbr_fea_len,
                                         atom_fea_len=args.atom_fea_len,
                                         n_conv=args.n_conv,
                                         h_fea_len=args.h_fea_len,
                                         n_h=args.n_h)
    else:
        classification = (args.task == 'classification')
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                   atom_fea_len=args.atom_fea_len,
                                   n_conv=args.n_conv,
                                   h_fea_len=args.h_fea_len,
                                   n_h=args.n_h,
                                   classification=classification)

    if args.cuda:
        model.cuda()

    # 定义损失函数
    if args.task == 'regression':
        criterion = nn.MSELoss()
    elif args.task == 'classification':
        criterion = nn.NLLLoss()
    else:  # multi-task
        criterion_reg = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()

    # 定义优化器
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # 学习率调度器
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    # 可选择载入预训练模型
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 载入检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> 载入检查点 '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> 未找到检查点 '{args.resume}'")

    # 训练和验证
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        train(train_loader, model, criterion, optimizer, epoch, args)

        # 验证
        if args.task == 'multi':
            mae_error = validate_multi(val_loader, model, criterion_reg, criterion_cls, args)
        else:
            mae_error = validate(val_loader, model, criterion, args)

        scheduler.step()

        # 记住最佳验证结果
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # 测试最佳模型
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    if args.task == 'multi':
        validate_multi(test_loader, model, criterion_reg, criterion_cls, args, test=True)
    else:
        validate(test_loader, model, criterion, args, test=True)


def train(train_loader, model, criterion, optimizer, epoch, args):
    """训练函数"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    if args.task == 'regression':
        mae_errors = AverageMeter()
    elif args.task == 'classification':
        accuracies = AverageMeter()
    else:  # multi-task
        mae_errors = AverageMeter()
        accuracies = AverageMeter()

    # 切换到训练模式
    model.train()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(train_loader):
        # 测量数据加载时间
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])

        # 前向传播
        if args.task == 'multi':
            target_reg = Variable(target[0])
            target_cls = Variable(target[1].squeeze())
            if args.cuda:
                target_reg = target_reg.cuda(non_blocking=True)
                target_cls = target_cls.cuda(non_blocking=True)
            
            output_reg, output_cls = model(*input_var)
            loss_reg = criterion_reg(output_reg, target_reg)
            loss_cls = criterion_cls(output_cls.squeeze(), target_cls)
            loss = loss_reg + args.cls_weight * loss_cls
        else:
            if args.task == 'regression':
                target_var = Variable(target.cuda(non_blocking=True) if args.cuda else target)
            else:
                target_var = Variable(target.cuda(non_blocking=True) if args.cuda else target)
            
            output = model(*input_var)
            loss = criterion(output, target_var)

        # 测量准确率和记录损失
        losses.update(loss.data.cpu().item(), target.size(0))
        
        if args.task == 'regression':
            mae_error = mae(output.data.cpu(), target)
            mae_errors.update(mae_error, target.size(0))
        elif args.task == 'classification':
            accuracy, _, _, _, _ = class_eval(output.data.cpu(), target)
            accuracies.update(accuracy, target.size(0))
        else:  # multi-task
            mae_error = mae(output_reg.data.cpu(), target[0])
            mae_errors.update(mae_error, target[0].size(0))
            accuracy = (output_cls.argmax(dim=1) == target_cls).float().mean()
            accuracies.update(accuracy.item(), target[1].size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 测量运行时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')
            elif args.task == 'classification':
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Accuracy {accuracies.val:.3f} ({accuracies.avg:.3f})')


def validate(val_loader, model, criterion, args, test=False):
    """验证函数"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # 切换到评估模式
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        
        if args.task == 'regression':
            target_var = Variable(target.cuda(non_blocking=True) if args.cuda else target)
        else:
            target_var = Variable(target.cuda(non_blocking=True) if args.cuda else target)

        # 前向传播
        with torch.no_grad():
            output = model(*input_var)
            loss = criterion(output, target_var)

        # 测量准确率和记录损失
        losses.update(loss.data.cpu().item(), target.size(0))
        
        if args.task == 'regression':
            mae_error = mae(output.data.cpu(), target)
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(output.data.cpu(), target)
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # 测量运行时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')
            else:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Accuracy {accuracies.val:.3f} ({accuracies.avg:.3f})')

    if test:
        star_label = '***'
    else:
        star_label = '*'

    if args.task == 'regression':
        print(f' {star_label} MAE {mae_errors.avg:.3f}')
        return mae_errors.avg
    else:
        print(f' {star_label} AUC {auc_scores.avg:.3f} Accuracy {accuracies.avg:.3f}'
              f' Precision {precisions.avg:.3f} Recall {recalls.avg:.3f}'
              f' F1 {fscores.avg:.3f}')
        return auc_scores.avg


def validate_multi(val_loader, model, criterion_reg, criterion_cls, args, test=False):
    """多任务验证函数"""
    batch_time = AverageMeter()
    losses_reg = AverageMeter()
    losses_cls = AverageMeter()
    mae_errors = AverageMeter()
    accuracies = AverageMeter()

    # 切换到评估模式
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        
        target_reg = Variable(target[0])
        target_cls = Variable(target[1].squeeze())
        if args.cuda:
            target_reg = target_reg.cuda(non_blocking=True)
            target_cls = target_cls.cuda(non_blocking=True)

        # 前向传播
        with torch.no_grad():
            output_reg, output_cls = model(*input_var)
            loss_reg = criterion_reg(output_reg, target_reg)
            loss_cls = criterion_cls(output_cls.squeeze(), target_cls)

        # 记录损失和准确率
        losses_reg.update(loss_reg.data.cpu().item(), target[0].size(0))
        losses_cls.update(loss_cls.data.cpu().item(), target[1].size(0))
        
        mae_error = mae(output_reg.data.cpu(), target[0])
        mae_errors.update(mae_error, target[0].size(0))
        
        accuracy = (output_cls.argmax(dim=1) == target_cls).float().mean()
        accuracies.update(accuracy.item(), target[1].size(0))

        # 测量运行时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Reg Loss {losses_reg.val:.4f} ({losses_reg.avg:.4f})\t'
                  f'Cls Loss {losses_cls.val:.4f} ({losses_cls.avg:.4f})\t'
                  f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                  f'Accuracy {accuracies.val:.3f} ({accuracies.avg:.3f})')

    if test:
        star_label = '***'
    else:
        star_label = '*'

    print(f' {star_label} MAE {mae_errors.avg:.3f} Accuracy {accuracies.avg:.3f}')
    return mae_errors.avg


class AverageMeter(object):
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """保存检查点"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def mae(prediction, target):
    """计算平均绝对误差"""
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    """分类评估"""
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


def get_train_val_test_loader(dataset, collate_fn, batch_size,
                              train_ratio, num_workers=1, val_ratio=0.1,
                              test_ratio=0.1, pin_memory=False,
                              return_test=False):
    """获取训练、验证和测试数据加载器"""
    total_size = len(dataset)
    indices = list(range(total_size))
    
    if train_ratio + val_ratio + test_ratio <= 1:
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size) if return_test else int((val_ratio + test_ratio) * total_size)
        test_size = total_size - train_size - val_size
    else:
        raise ValueError('train_ratio + val_ratio + test_ratio should not exceed 1.')

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    if return_test:
        test_indices = indices[train_size + val_size:]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    if return_test:
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    main() 