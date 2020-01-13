import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred.cpu(), dim=-1), 1))


def flip_along_batch(input, step=-1):
    inv_idx = torch.arange(input.size(0) - 1, -1, step).long()
    return input[inv_idx]


def label_smoothing(pred, target, eta=0.1):
    n_classes = pred.size(1)
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_classes * 1


class CrossEntropyWithLabelSmoothing(nn.Module):
    def __init__(self):
        super(CrossEntropyWithLabelSmoothing, self).__init__()

    def forward(self, pred, target, eta=0.1):
        onehot_target = label_smoothing(pred, target, eta=eta)
        return cross_entropy_for_onehot(pred, onehot_target)


def mixup_data(inputs, lam=1):
    inputs = np.asanyarray(inputs)
    flipped_inputs = inputs[::-1]
    flipped_inputs = torch.tensor(np.asarray(flipped_inputs))
    return lam * inputs + (1 - lam) * flipped_inputs


def mixup_label(pred, target, lam=1, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta=eta)
    onehot_target = np.asarray(onehot_target.cpu())
    flipped_target = onehot_target[::-1]
    return torch.tensor(lam * onehot_target + (1 - lam) * flipped_target)


class CrossEntropyWithMixup(nn.Module):
    def __init__(self):
        super(CrossEntropyWithMixup, self).__init__()

    def forward(self, pred, target, lam=1, eta=0.0):
        mixup_target = mixup_label(pred, target, lam=lam, eta=eta)
        return cross_entropy_for_onehot(pred, mixup_target)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_one_epoch(logger, scheduler, train_loader, model, criterion, optimizer, use_cuda,
                    low_precision_training, label_smoothing, mixup):
    model.train()
    losses = []
    accs = []
    normal_scheduler = False
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        if label_smoothing > 0 and mixup == 1:
            loss = criterion(outputs, targets, label_smoothing)
        elif label_smoothing > 0 and mixup < 1:
            loss = criterion(outputs, targets, mixup, 0)
        elif mixup < 1 and not label_smoothing == 0:
            loss = criterion(outputs, targets, mixup, label_smoothing)
        else:
            loss = criterion(outputs, targets)

        logger.train_losses.append(loss.item())
        losses.append(loss.item())
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 2))
        logger.train_accs.append(prec1)
        accs.append(prec1.item())
        optimizer.zero_grad()

        try:
            scheduler.step()
        except:
            normal_scheduler = True
        if low_precision_training:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
    if normal_scheduler:
        scheduler.step(prec1)
    return np.mean(losses), np.mean(accs)
