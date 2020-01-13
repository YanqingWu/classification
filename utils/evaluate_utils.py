import numpy as np
from utils.train_utils import accuracy


def val_one_epoch(logger, val_loader, model, criterion, use_cuda):
    model.eval()
    losses = []
    accs = []
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 2))
        losses.append(loss.item())
        accs.append(prec1.item())
    logger.val_losses.extend(losses)
    logger.val_accs.extend(accs)
    return np.mean(losses), np.mean(accs)