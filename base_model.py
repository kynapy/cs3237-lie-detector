import os
import shutil
import time
import platform
from enum import Enum
import torch
from torch import nn, optim, distributed
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from meterbar import Summary, AverageMeter, ProgressMeter

DATA_DIR = "./affectnet_hq/archive"
RESUME_PATH = "./base_model.pth.tar"
NUM_CLASSES = 8
BATCH_SIZE = 64
NUM_EPOCES = 30


def load_dataset(data_dir, val_split):
    all_data = datasets.ImageFolder(
        data_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=.3, hue=.3, contrast=.3, saturation=.3),
            transforms.RandomEqualize(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    val_size = int(val_split * len(all_data))
    train_size = len(all_data) - val_size
    train_dataset, val_dataset = random_split(all_data, [train_size, val_size])
    return train_dataset, val_dataset

class BaseModel(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super().__init__()
        weights = "DEFAULT" if pretrain else None
        self.model = models.regnet_y_400mf(weights=weights)
        self.model.fc = nn.Linear(440, num_classes)

    def forward(self, img):
        return self.model(img)


def main_worker(data_dir, num_classes, batch_size, epochs, resume=None, workers=0):
    model = BaseModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), 3e-4, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_dataset, val_dataset = load_dataset(data_dir, 0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    if resume and os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found, start a new model.")
        start_epoch = 0
        best_acc1 = 0

    for epoch in range(start_epoch, epochs):
        train(train_loader, model, criterion, optimizer, epoch, device)
        acc1 = validate(val_loader, model, criterion)
        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, resume, is_best)
        torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=200):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1, ))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, print_freq=80):

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target, topk=(1, ))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, filename, is_best=False):
    torch.save(state, filename)
    if is_best:
        best_path = os.path.splitext(os.path.splitext(filename)[0])[0] + "_best.pth.tar"
        shutil.copyfile(filename, best_path)


def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Using GPU.")
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        print("Using CPU.")
        device = torch.device("cpu")
    workers = 2 if platform.system() == "Linux" else 0
    main_worker(DATA_DIR, NUM_CLASSES, BATCH_SIZE, NUM_EPOCES, RESUME_PATH, workers)
