import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

""" load data args"""
parser.add_argument('-d', '--data', default='data', type=str,
                    help='data root path, have {train, val} under root, '
                         'every class is a single folder under {train, val}.')

parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers, windows need to change to 0.')

""" lr args """
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, dest='lr',
                    help='initial learning rate')

parser.add_argument('-wp', '--warmup', action='store_true',
                    help='learning rate warmup')

parser.add_argument('-we', '--warmup-epochs', default=10, type=int,
                    help='learning rate warmup epochs, only if warmup is true')

parser.add_argument('-ls', '--linear-scaling', action='store_true',
                    help='linear scaling learning rate')

parser.add_argument('-cos', '--cosine', action='store_true',
                    help='cosine learning rate decay')

""" low precision training  """
parser.add_argument('-lpt', '--low-precision-training', action='store_true',
                    help='low precision training')

""" model args """
parser.add_argument('-a', '--arch', default='',
                    help='model architecture')

""" train args """
parser.add_argument('-ep', '--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-tb', '--train-batch', default=16, type=int,
                    help='train batch size')

parser.add_argument('-p', '--pretrained', action='store_false',
                    help='use pretrained model')

""" label smoothing """
parser.add_argument('-lst', '--label-smoothing', default=0.0,
                    help='use label smoothing')

""" mixup training """
parser.add_argument('-mixup', '--mixup', default=1.0,
                    help='use mixup training')

""" optimizer """
parser.add_argument('-m', '-momentum', default=0.9, type=float, dest='momentum',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay for all parameters, if No bias decay is set False.')
parser.add_argument('-nb', '--no-bias-decay', action='store_true',
                    help='no bias decay')

""" resume """
parser.add_argument('-r', '--resume', type=str,
                    help='resume model')

""" val args """
parser.add_argument('-vb', '--val-batch', default=8, type=int,
                    help='val batch size')

""" seed args """
parser.add_argument('-seed', '--manual-seed', type=int,
                    help='manual seed')

""" gpu args """
parser.add_argument('-ug', '--use-gpu', action='store_true',
                    help='use gpu training')

parser.add_argument('-sgi', '--single-gpu-id', type=int, default=0,
                    help='gpu id for training')

parser.add_argument('-mg', '--multi-gpus', action='store_true',
                    help='multi gpu training')

""" logs """
parser.add_argument('-log_file', '--log_file_path', default='logs',
                    help='where to save logs')

