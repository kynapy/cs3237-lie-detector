import os
import shutil
import time
import random
import platform
from enum import Enum
import numpy as np
import torch
from torch import nn, optim, distributed
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.nn.utils import rnn
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import KFold

DATA_DIR = "./second_step"
BASE_MODEL_PATH = "./fer_base_model_best.pth.tar"
RESUME_PATH = "./predict_model.pth.tar"
NUM_CLASSES = 8
BATCH_SIZE = 4
NUM_EPOCES = 10
PRINT_FREQ = 5


class Model(nn.Module):

    def __init__(self, base_classes, hidden_dim=64, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.base_model = models.regnet_y_1_6gf(weights=None)
        self.base_model.fc = nn.Linear(888, base_classes)
        self.lstm = nn.LSTM(input_size=base_classes + 1, hidden_size=hidden_dim, num_layers=lstm_layers, dropout=dropout, batch_first=True)
        self.dense = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, imgs, rates, seq_len):
        img_vecs = [self.base_model(img[:seq_len[i]]) for (i, img) in enumerate(imgs)]
        pad_imgvecs = rnn.pad_sequence(img_vecs, batch_first=True, padding_value=0)
        conbine = torch.concat((pad_imgvecs, rates.unsqueeze(2)), dim=2)
        conbine = rnn.pack_padded_sequence(conbine, seq_len, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.lstm(conbine)
        lstm_out, output_lens = rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return self.dense(lstm_out[:, -1, :])

    def load_base(self, base_state=None):
        if base_state:
            checkpoint = torch.load(base_state)
            self.base_model.load_state_dict(checkpoint["state_dict"])


def load_dataset(data_dir, valid_split, seed=3237):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=.3, hue=.3, contrast=.3, saturation=.3),
        transforms.RandomEqualize(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    srcs, tgts = [], []
    categories = os.listdir(data_dir)
    length = 0
    for i, cat in enumerate(categories, 0):
        cat_base = os.path.join(DATA_DIR, cat)
        questions = [os.path.join(cat_base, folder) for folder in os.listdir(cat_base)]
        for q in questions:
            imgfiles = [f for f in os.listdir(q) if f.endswith(".png")]
            imgs = torch.tensor(np.array([transform(Image.open(os.path.join(q, f))).numpy() for f in imgfiles]))
            with open(os.path.join(q, "hrData.txt")) as hrfile:
                numbers = [s.split(" : ") for s in hrfile.readlines()]
                assert not [n for n in numbers if len(n) == 3]
                timestamps = [os.path.splitext(f)[0].lstrip("image_") for f in imgfiles]
                hrdata = [float(s[1]) for s in numbers if s[0].lstrip("(").rstrip(")").replace(":", "_") in timestamps]
            assert len(imgs) == len(hrdata)
            srcs.append((imgs, torch.tensor(hrdata)))
            tgts.append(i)
            length += 1
    random.seed(seed)
    random.shuffle(srcs)
    random.seed(seed)
    random.shuffle(tgts)
    valid_len = max(int(length * valid_split), 1)
    train_len = length - valid_len
    return (srcs[:train_len], tgts[:train_len]), (srcs[train_len:], tgts[train_len:])


class MyData(data.Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.srcs, self.tgts = dataset
        self.tgts = [torch.tensor([n], dtype=torch.float) for n in self.tgts]
        self.len = len(self.tgts)

    def __getitem__(self, i: int):
        return self.srcs[i], self.tgts[i]

    def __len__(self):
        return self.len


def padding(data):
    imgs = [f[0][0] for f in data]
    hrdata = [f[0][1] for f in data]
    target = [f[1] for f in data]
    length = [len(d) for d in hrdata]
    pad_imgs = rnn.pad_sequence(imgs, batch_first=True, padding_value=0)
    pad_hr = rnn.pad_sequence(hrdata, batch_first=True, padding_value=0)
    return (pad_imgs, pad_hr, torch.tensor(target)), length


def main_worker(data_dir, num_classes, batch_size, epochs, base_resume=None, resume=None, workers=0):
    model = Model(num_classes).to(device)
    model.load_base(base_resume)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), 2e-5, weight_decay=3e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_data, val_data = load_dataset(data_dir, 0.2)
    train_loader = data.DataLoader(MyData(train_data), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=padding)
    val_loader = data.DataLoader(MyData(val_data), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=padding)

    if resume and os.path.isfile(resume):
        print("=> loading checkpoint {}".format(resume))
        checkpoint = torch.load(resume, map_location=device)
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("=> loaded checkpoint {} (epoch {})".format(resume, checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at", resume)
        start_epoch = 0
        best_acc1 = 0

    for epoch in range(start_epoch, epochs):
        train(train_loader, model, criterion, optimizer, epoch, device)
        acc1 = validate(val_loader, model, criterion)
        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, resume, is_best)
        torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=PRINT_FREQ):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (batch, length) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, rates, target = batch

        images = images.to(device, non_blocking=True)
        rates = rates.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images, rates, length)
        loss = criterion(output, target.unsqueeze(1))

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


def validate(val_loader, model, criterion, print_freq=PRINT_FREQ):

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix="Test: ")

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (batch, length) in enumerate(val_loader):
            images, rates, target = batch

            images = images.to(device, non_blocking=True)
            rates = rates.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images, rates, length)
            loss = criterion(output, target.unsqueeze(1))

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
        best_path = os.path.splitext(os.path.splitext(filename)[0])[0] + "_new.pth.tar"
        shutil.copyfile(filename, best_path)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        distributed.all_reduce(total, distributed.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


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


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using GPU.")
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        print("Using CPU.")
        device = torch.device("cpu")
    workers = 2 if platform.system() == "Linux" else 0
    main_worker(DATA_DIR, NUM_CLASSES, BATCH_SIZE, NUM_EPOCES, BASE_MODEL_PATH, RESUME_PATH, workers)
