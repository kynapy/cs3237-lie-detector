import os
import shutil
import time
import random
import platform
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.nn.utils import rnn
from torchvision import models, transforms
from PIL import Image

from meterbar import Summary, AverageMeter, ProgressMeter
from optimal_flow import face_landmarks, dense_flow

DATASETS_DIR = "./second_data"
BASE_MODEL_PATH = "./base_model.pth.tar"
RESUME_PATH = "./predict_model.pth.tar"
NUM_CLASSES = 8
NUM_FRAMES = 9
BATCH_SIZE = 4
NUM_EPOCES = 10
PRINT_FREQ = 10
KFOLD = 4


class ImgModel(nn.Module):

    def __init__(self, num_classes, weights=None):
        super().__init__()
        if weights:
            base_model = models.regnet_y_400mf(weights=None)
            base_model.fc = nn.Linear(440, num_classes)
            checkpoint = torch.load(weights)
            base_model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            base_model = models.regnet_y_400mf(weights="DEFAULT")
            base_model.fc = nn.Linear(440, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)


class FlowModel(nn.Module):

    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.model = models.regnet_y_400mf(weights="DEFAULT")
        self.model.stem = nn.Sequential(nn.Conv2d(in_channel, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True))
        self.model.fc = nn.Linear(440, num_classes)

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):

    def __init__(self, num_classes, num_frames, weights=None, hidden_size1=64, hidden_size2=8):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.img_model = ImgModel(num_classes, weights)
        self.flow_model = FlowModel(3 * num_frames - 2, hidden_size1)
        self.lstm1 = nn.LSTM(input_size=num_classes, hidden_size=hidden_size1, num_layers=3, dropout=0.5, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=hidden_size2, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Sequential(nn.Linear(3 * hidden_size1 + 2 * hidden_size2, 1), nn.Sigmoid())
        self.landmarks = face_landmarks()
        self.denseflow = dense_flow()
        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def forward(self, imgs, rates, seq_len):
        img_vecs, flows = [], []
        for (i, img) in enumerate(imgs):
            sample = uniform_choose(range(seq_len[i]), self.num_frames)
            frames = [img[j].clone().detach().cpu().numpy() for j in sample]
            flow = []
            for j in range(len(frames)):
                landmark = self.landmarks(frames[j])
                markmat = np.expand_dims(np.zeros_like(frames[j][:,:,0]), 0)
                for x, y in landmark:
                    markmat[0][x][y] = 1
                flow.append(markmat)
                if j > 0:
                    flow.append(self.denseflow(frames[j - 1], frames[j], "RLOF"))
            flow = np.concatenate(flow, axis=0)
            flows.append(flow)

            norm_image = []
            for l in range(seq_len[i]):
                norm_image.append(self.norm(img[l].clone().detach().cpu().numpy()).numpy())
            norm_image = torch.tensor(np.array(norm_image), dtype=torch.float32, device=device)
            img_vecs.append(self.img_model(norm_image))

        pad_img = rnn.pad_sequence(img_vecs, batch_first=True, padding_value=0)
        pack_img = rnn.pack_padded_sequence(pad_img, seq_len, batch_first=True, enforce_sorted=False)
        img_out, _ = self.lstm1(pack_img)
        img_out, _ = rnn.pad_packed_sequence(img_out, batch_first=True)

        flows = np.array(flows)
        flows_vec = self.flow_model(torch.tensor(flows, dtype=torch.float32, device=device))

        hr_packed = rnn.pack_padded_sequence(rates.unsqueeze(2), seq_len, batch_first=True, enforce_sorted=False)
        hr_out, _ = self.lstm2(hr_packed)
        hr_out, _ = rnn.pad_packed_sequence(hr_out, batch_first=True)

        out = torch.concat((img_out[:, -1, :], flows_vec, hr_out[:, -1, :]), dim=1)
        return self.dense(out)


def load_dataset(data_dir, seed=3237):
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    srcs, tgts = [], []
    datasets = [os.path.join(data_dir, dir) for dir in os.listdir(data_dir) if not dir.endswith(".zip")]
    for dataset in datasets:
        lies = os.path.join(dataset, "lie")
        truths = os.path.join(dataset, "truth")
        questions = [os.path.join(lies, folder) for folder in os.listdir(lies)]
        for q in questions:
            imgfiles = [f for f in os.listdir(q) if f.endswith(".png")]
            imgs = np.array([np.array(transform(Image.open(os.path.join(q, f)))) for f in imgfiles])
            with open(os.path.join(q, "hrData.txt")) as hrfile:
                hrdata = np.array([float(line.split()[-1]) for line in hrfile.readlines()])
            try:
                assert len(imgs) == len(hrdata)
            except:
                print("Not the same length:", q)
            else:
                srcs.append((torch.tensor(imgs, dtype=torch.uint8), torch.tensor(hrdata, dtype=torch.float32)))
                tgts.append(0)
        questions = [os.path.join(truths, folder) for folder in os.listdir(truths)]
        for q in questions:
            imgfiles = [f for f in os.listdir(q) if f.endswith(".png")]
            imgs = np.array([np.array(transform(Image.open(os.path.join(q, f)))) for f in imgfiles])
            with open(os.path.join(q, "hrData.txt")) as hrfile:
                hrdata = np.array([float(line.split()[-1]) for line in hrfile.readlines()])
            try:
                assert len(imgs) == len(hrdata)
            except:
                print("Not the same length:", q)
            else:
                srcs.append((torch.tensor(imgs, dtype=torch.uint8), torch.tensor(hrdata, dtype=torch.float32)))
                tgts.append(1)
    random.seed(seed)
    random.shuffle(srcs)
    random.seed(seed)
    random.shuffle(tgts)
    return srcs, tgts


def Kfold(all_data, K):
    srcs, tgts = all_data
    total = len(tgts)
    onefold = total / K
    print("Data loaded:", total)
    inter = [0] + [int(round(onefold * k)) for k in range(1, K)] + [total + 1]
    sfolds = [srcs[inter[i]:inter[i + 1]] for i in range(K)]
    tfolds = [tgts[inter[i]:inter[i + 1]] for i in range(K)]
    ret = []
    for i in range(K):
        strain, ttrain = [], []
        for j in range(K):
            if j != i:
                strain.extend(sfolds[j])
                ttrain.extend(tfolds[j])
        ret.append(((strain, ttrain), (sfolds[i], tfolds[i])))


def split_data(all_data, k):
    srcs, tgts = all_data
    total = len(tgts)
    val_len = int(k * total)
    train_len = total - val_len
    print("Data loaded: train %d, validation %d" % (train_len, val_len))
    return (srcs[:train_len], tgts[:train_len]), (srcs[train_len:], tgts[train_len:])


def uniform_choose(iterable, num_sample):
    l = len(iterable)
    i, interval = 0, l / num_sample
    choice = []
    for _ in range(num_sample):
        choice.append(iterable[int(round(i))])
        i += interval
    return choice


class MyData(data.Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.srcs, self.tgts = dataset
        self.tgts = [torch.tensor([n], dtype=torch.float) for n in self.tgts]
        self.len = len(self.tgts)

    def __getitem__(self, i: int):
        # transform = transforms.Compose([
        # transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.ColorJitter(brightness=.3, hue=.3, contrast=.3, saturation=.3),
        # transforms.RandomEqualize(),
        # transforms.RandomHorizontalFlip(),
        #     transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        # ])
        # origin_img, hrdata = self.srcs[i]
        # new_img = torch.tensor(np.array([transform(img).numpy() for img in origin_img]))
        # return (new_img, hrdata), self.tgts[i]
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


def main_worker(data_dir, num_classes, num_frames, batch_size, epochs, base_resume=None, resume=None, workers=0, K=0):
    model = Model(num_classes, num_frames, base_resume).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), 3e-4, weight_decay=3e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    all_data = load_dataset(data_dir, seed=1)
    train_data, val_data = split_data(all_data, 0.25)
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


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using GPU.")
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        print("Using CPU.")
        device = torch.device("cpu")
    workers = 2 if platform.system() == "Linux" else 0
    main_worker(DATASETS_DIR, NUM_CLASSES, NUM_FRAMES, BATCH_SIZE, NUM_EPOCES, BASE_MODEL_PATH, RESUME_PATH, workers, KFOLD)
