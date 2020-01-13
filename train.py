import time
import random
import warnings
import numpy as np
from models.dla import *
import torch.nn.parallel
import torch.optim as optim
from utils.logger import logger
from config.config import parser
from cnn_finetune import make_model
from utils.data_loader import DataLoader
from models.efficientnet import EfficientNet
from utils.evaluate_utils import val_one_epoch
from utils.train_utils import train_one_epoch, CrossEntropyWithLabelSmoothing, CrossEntropyWithMixup

warnings.filterwarnings('ignore')


def main():
    args = parser.parse_args()
    log = logger(args)
    log.write('V' * 50 + " configs " + 'V' * 50 + '\n')
    log.write(args)
    log.write('')
    log.write('Λ' * 50 + " configs " + 'Λ' * 50 + '\n')

    # load data
    input_size = (224, 224)
    dataset = DataLoader(args, input_size)
    train_data, val_data = dataset.load_data()

    num_classes = dataset.num_classes
    classes = dataset.classes
    log.write('\n\n')
    log.write('V' * 50 + " data " + 'V' * 50 + '\n')
    log.info('success load data.')
    log.info('num classes: %s' % num_classes)
    log.info('classes: ' + str(classes) + '\n')
    log.write('Λ' * 50 + " data " + 'Λ' * 50 + '\n')

    # Random seed
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    log.write('random seed is %s' % args.manual_seed)

    # pretrained or not
    log.write('\n\n')
    log.write('V' * 50 + " model " + 'V' * 50 + '\n')
    if args.pretrained:
        log.info("using pre-trained model")
    else:
        log.info("creating model from initial")

    # model
    log.info('using model: %s' % args.arch)
    log.write('')
    log.write('Λ' * 50 + " model " + 'Λ' * 50 + '\n')

    # resume model
    if args.resume:
        log.info('using resume model: %s' % args.resume)
        states = torch.load(args.resume)
        model = states['model']
        model.load_state_dict(states['state_dict'])
    else:
        log.info('not using resume model')
        if args.arch.startswith('dla'):
            model = eval(args.arch)(args.pretrained, num_classes)

        elif args.arch.startswith('efficientnet'):
            if args.pretrained:
                model = EfficientNet.from_pretrained(args.arch, num_classes=num_classes)
            else:
                model = EfficientNet.from_name(args.arch, num_classes=num_classes)
        else:
            model = make_model(model_name=args.arch,
                               num_classes=num_classes,
                               pretrained=args.pretrained,
                               pool=nn.AdaptiveAvgPool2d(output_size=1),
                               classifier_factory=None,
                               input_size=input_size,
                               original_model_state_dict=None,
                               catch_output_size_exception=True)

    # cuda
    have_cuda = torch.cuda.is_available()
    use_cuda = args.use_gpu and have_cuda
    log.info('using cuda: %s' % use_cuda)
    if have_cuda and not use_cuda:
        log.info('\nWARNING: found gpu but not use, you can switch it on by: -ug or --use-gpu\n')

    multi_gpus = False
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if args.multi_gpus:
            gpus = torch.cuda.device_count()
            multi_gpus = gpus > 1

    if multi_gpus:
        log.info('using multi gpus, found %d gpus.' % gpus)
        model = torch.nn.DataParallel(model).cuda()
    elif use_cuda:
        model = model.cuda()

    # criterian
    log.write('\n\n')
    log.write('V' * 50 + " criterion " + 'V' * 50 + '\n')
    if args.label_smoothing > 0 and args.mixup == 1:
        criterion = CrossEntropyWithLabelSmoothing()
        log.info('using label smoothing criterion')

    elif args.label_smoothing > 0 and args.mixup < 1:
        criterion = CrossEntropyWithMixup()
        log.info('using label smoothing and mixup criterion')

    elif args.mixup < 1 and not args.label_smoothing == 0:
        criterion = CrossEntropyWithMixup()
        log.info('using mixup criterion')

    else:
        criterion = nn.CrossEntropyLoss()
        log.info('using normal cross entropy criterion')

    if use_cuda:
        criterion = criterion.cuda()

    log.write('using criterion: %s' % str(criterion))
    log.write('')
    log.write('Λ' * 50 + " criterion " + 'Λ' * 50 + '\n')
    # optimizer
    log.write('\n\n')
    log.write('V' * 50 + " optimizer " + 'V' * 50 + '\n')
    if args.linear_scaling:
        args.lr = 0.1 * args.train_batch / 256
    log.write('initial lr: %4f\n' % args.lr)
    if args.no_bias_decay:
        log.info('using no bias weight decay')
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]
        optimizer = optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum)

    else:
        log.info('using bias weight decay')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        optimizer.load_state_dict(states['optimizer'])
    log.write('using optimizer: %s' % str(optimizer))
    log.write('')
    log.write('Λ' * 50 + " optimizer " + 'Λ' * 50 + '\n')
    # low precision
    use_low_precision_training = args.low_precision_training
    if use_low_precision_training:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # lr scheduler
    iters_per_epoch = int(np.ceil(len(train_data) / args.train_batch))
    total_iters = iters_per_epoch * args.epochs
    log.write('\n\n')
    log.write('V' * 50 + " lr_scheduler " + 'V' * 50 + '\n')
    if args.warmup:
        log.info('using warmup scheduler, warmup epochs: %d' % args.warmup_epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   iters_per_epoch * args.warmup_epochs, eta_min=1e-6)
    elif args.cosine:
        log.info('using cosine lr scheduler')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    else:
        log.info('using normal lr decay scheduler')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6, mode='min')

    log.write('using lr scheduler: %s' % str(scheduler))
    log.write('')
    log.write('Λ' * 50 + " lr_scheduler " + 'Λ' * 50 + '\n')

    log.write('\n\n')
    log.write('V' * 50 + " training start " + 'V' * 50 + '\n')
    best_acc = 0
    start = time.time()
    log.info('\nstart training ...')
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[-1]['lr']
        train_loss, train_acc = train_one_epoch(log, scheduler, train_data, model, criterion, optimizer,
                                                use_cuda, use_low_precision_training,
                                                args.label_smoothing, args.mixup)
        test_loss, test_acc = val_one_epoch(log, val_data, model, criterion, use_cuda)
        end = time.time()
        log.info('epoch: [%d / %d], time spent(s): %.2f, mean time: %.2f, lr: %.4f, train loss: %.4f, train acc: %.4f, '
                 'test loss: %.4f, test acc: %.4f' %
                 (epoch, args.epochs, end-start, (end-start) / epoch, lr, train_loss, train_acc, test_loss, test_acc))
        states = dict()
        states['arch'] = args.arch
        if multi_gpus:
            states['model'] = model.module
            states['state_dict'] = model.module.state_dict()
        else:
            states['model'] = model
            states['state_dict'] = model.state_dict()
        states['optimizer'] = optimizer.state_dict()
        states['test_acc'] = test_acc
        states['train_acc'] = train_acc
        states['epoch'] = epoch
        states['classes'] = classes
        is_best = False
        if test_acc > best_acc:
            is_best = True
            log.save_checkpoint(states, is_best)
        else:
            log.save_checkpoint(states, is_best)

    log.write('\ntraining finished.')
    log.write('Λ' * 50 + " training finished " + 'Λ' * 50 + '\n')
    log.log_file.close()
    log.writer.close()


if __name__ == "__main__":
    main()
