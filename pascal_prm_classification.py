import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
#from tensorboardX import SummaryWriter
import ipdb
from voc_dataset import Voc2007Classification
import torch.nn.functional as F
from peak_stimulation import peak_stimulation

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
	     'bottle', 'bus', 'car', 'cat', 'chair',
	     'cow', 'diningtable', 'dog', 'horse',
	     'motorbike', 'person', 'pottedplant',
	     'sheep', 'sofa', 'train', 'tvmonitor')


def multilabel_soft_margin_loss(
    input,
    target,
    weight = None,
    size_average = True,
    reduce = True,
    difficult_samples = False):
    """Multilabel soft margin loss.
    """

    if difficult_samples:
        # label 1: positive samples
        # label 0: difficult samples
        # label -1: negative samples
        gt_label = target.clone()
        gt_label[gt_label == 0] = 1
        gt_label[gt_label == -1] = 0
    else:
        gt_label = target
        
    return F.multilabel_soft_margin_loss(input, gt_label, weight, size_average, reduce)


def median_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_ResNet, self).__init__()

        # feature encoding
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classifier
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def fc_resnet50(num_classes=20, pretrained=True):
    """FC ResNet50.
    """
    model = FC_ResNet(models.resnet50(pretrained), num_classes)
    return model


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dir_datasets', default='/export/home/ehsan/data/dataset/detection/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-n', '--name', default='default', type=str,
                    help='checkpoint directory')
parser.add_argument('-ims', '--image_size', default=224, type=int,
                    help='image size')

best_acc1 = 0


def main():
    args = parser.parse_args()
    #log_dir = os.path.join('log/recognition/{}'.format(args.name))
    #os.makedirs(log_dir, exist_ok=True)
    #writer = SummaryWriter()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        #main_worker(args.gpu, ngpus_per_node, args, writer, log_dir)
        main_worker(args.gpu, ngpus_per_node, args)


#def main_worker(gpu, ngpus_per_node, args, writer, logdir):
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        sys.stdout.flush()

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        sys.stdout.flush()
        
        if args.arch == 'fc_resnet50':
            model = fc_resnet50(num_classes=20, pretrained=True)

        else:
            model = models.__dict__[args.arch](pretrained=True)
            if args.arch in ['resnet18']:    
                model.fc = torch.nn.Linear(512, len(CLASSES))
            elif 'squeezenet' in args.arch:
                model.classifier[1] = nn.Conv2d(512, len(CLASSES), (1,1), (1,1))
                model.num_classes = len(CLASSES)
            elif 'resnet' in args.arch:
                model.fc = torch.nn.Linear(2048, len(CLASSES))
    else:
        print("=> creating model '{}'".format(args.arch))
        sys.stdout.flush()
        model = models.__dict__[args.arch](num_classes=len(CLASSES))
    #ipdb.set_trace()
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    sys.stdout.flush()
    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #train_dataset = datasets.ImageFolder(
    #    traindir,
    #    transforms.Compose([
    #        transforms.RandomResizedCrop(224),
    #        #transforms.Resize((224,224)),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))

    img_transform = transforms.Compose([
            #transforms.RandomResizedCrop(448),
            transforms.Resize((args.image_size,args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    train_dataset = Voc2007Classification(args.dir_datasets, 'train', transform=img_transform)
    val_dataset = Voc2007Classification(args.dir_datasets, 'val', transform=img_transform)
    test_dataset = Voc2007Classification(args.dir_datasets, 'test', transform=img_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    #val_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(valdir, transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])),
    #    batch_size=256, shuffle=False,
    #    num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        #validate(val_loader, model, criterion, args, writer=writer)
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch, args, writer, logdir)
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        #acc1 = validate(val_loader, model, criterion, args, writer, logdir)
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, name=args.name, size=args.image_size)


#def train(train_loader, model, criterion, optimizer, epoch, args, writer, logdir):
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, name, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        #new_target = (target==1).type(torch.float32)
        new_target = (target!=-1).type(torch.float32)
       
        # compute output
        output = model(input)
       
        if args.arch == 'fc_resnet50':
            batch, cl, _, _ = output.shape
            #output = output.view(batch, cl, -1).max(2)[0]
            #output = F.adaptive_avg_pool2d(output_size=(1), input=output).squeeze()
            peak_list, output = peak_stimulation(output, return_aggregation=True, win_size=3, peak_filter=median_filter)
            loss = multilabel_soft_margin_loss(output, target, difficult_samples=True)
        else:
            #loss = criterion(output, new_target)
            loss = multilabel_soft_margin_loss(output, target, difficult_samples=True)
       
        # measure accuracy and record loss
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy_multi(output, target, 0.5)
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        #top1.update(acc1[0], input.size(0))
        #top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            #writer.add_scalar(os.path.join(logdir, 'train_loss'.format(args.name)), losses.avg)
            #writer.add_scalar(os.path.join(logdir, 'train_acc'.format(args.name)), top1.avg)
            sys.stdout.flush()


#def validate(val_loader, model, criterion, args, writer, logdir):
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, _, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            #new_target = (target==1).type(torch.float32)
            new_target = (target!=-1).type(torch.float32)
            if args.arch == 'fc_resnet50':
                batch, cl, _, _ = output.shape
                #output = output.view(batch, cl, -1).max(2)[0]
                #output = F.adaptive_avg_pool2d(output_size=(1), input=output).squeeze()
                peak_list, output = peak_stimulation(output, return_aggregation=True, win_size=3, peak_filter=median_filter)
                loss = multilabel_soft_margin_loss(output, target, difficult_samples=True)
            else:
                #loss = criterion(output, new_target)
                loss = multilabel_soft_margin_loss(output, target, difficult_samples=True)

            acc1 = accuracy_multi(output, target, 0.5)
            # measure accuracy and record loss
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
            #top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))
                sys.stdout.flush()

        #print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #      .format(top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
        #writer.add_scalar(os.path.join(logdir, 'val_loss'.format(args.name)), losses.avg)
        #writer.add_scalar(os.path.join(logdir, 'val_acc'.format(args.name)), top1.avg)
        sys.stdout.flush()

    return top1.avg


def save_checkpoint(state, is_best, name, filename='checkpoint.pth.tar', size=224):
    savename = '{}_{}'.format(name, size)
    save_dir = os.path.join('checkpoints', savename)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, filename)
    save_best = os.path.join(save_dir, 'model_best.pth.tar')
    torch.save(state, save_file)
    if is_best:
        shutil.copyfile(save_file, save_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_multi(output, target, threshold):
    pred = (output.sigmoid()>threshold).type(torch.float32)
    target = (target==1).type(torch.float32)
    acc = (pred==target).type(torch.float32).view(1, -1).mean()
    return acc

if __name__ == '__main__':
    main()
