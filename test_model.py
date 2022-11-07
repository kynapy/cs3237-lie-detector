import os
import time
from PIL import Image
import numpy as np
import torch
from torch.backends import cudnn
from torch import nn, optim
from torch.nn.utils import rnn
from torchvision import models, transforms
from optimal_flow import BatchFlow

DATA_DIR = ""
MODEL_PATH = "./predict_model.pth"
CONTINUE_UPDATE = False
NUM_CLASSES = 8
NUM_FRAMES = 9


class ImgModel(nn.Module):

    def __init__(self, num_classes, weights="default"):
        super().__init__()
        if weights:
            if weights == "default":
                base_model = models.regnet_y_400mf(weights="DEFAULT")
                base_model.fc = nn.Linear(440, num_classes)
            else:
                base_model = models.regnet_y_400mf(weights=None)
                base_model.fc = nn.Linear(440, num_classes)
                checkpoint = torch.load(weights)
                base_model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            base_model = models.regnet_y_400mf(weights=None)
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

    def __init__(self, num_classes, num_frames, weights=None, hidden_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.hidden = hidden_dim
        self.img_model = ImgModel(num_classes, weights)
        self.flow_model = FlowModel(3 * num_frames - 2, hidden_dim)
        self.lstm1 = nn.LSTM(input_size=num_classes, hidden_size=hidden_dim, num_layers=3, dropout=0.5, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.dense = nn.Sequential(nn.Conv1d(5, 1, 1), nn.Flatten(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def forward(self, imgs, flows, rates, seq_len):
        img_vecs = []
        for (i, img) in enumerate(imgs):
            norm_image = []
            for l in range(seq_len[i]):
                norm_image.append(self.norm(img[l].clone().detach().cpu().numpy()).numpy())
            norm_image = torch.tensor(np.array(norm_image), dtype=torch.float32, device=device)
            img_vecs.append(self.img_model(norm_image))

        pad_img = rnn.pad_sequence(img_vecs, batch_first=True, padding_value=0)
        pack_img = rnn.pack_padded_sequence(pad_img, seq_len, batch_first=True, enforce_sorted=False)
        img_out, _ = self.lstm1(pack_img)
        img_out, _ = rnn.pad_packed_sequence(img_out, batch_first=True)

        flows_vec = self.flow_model(flows)

        hr_packed = rnn.pack_padded_sequence(rates.unsqueeze(2), seq_len, batch_first=True, enforce_sorted=False)
        hr_out, _ = self.lstm2(hr_packed)
        hr_out, _ = rnn.pad_packed_sequence(hr_out, batch_first=True)

        img_feature = img_out[:, -1, :].view(-1, 2, self.hidden)
        flow_feature = flows_vec.unsqueeze(1)
        hr_feature = hr_out[:, -1, :].view(-1, 2, self.hidden)
        out = torch.concat((img_feature, flow_feature, hr_feature), dim=1)
        return self.dense(out)


class LoadData(object):

    def __init__(self, num_frames):
        self.denseflow = BatchFlow(num_frames)
        self.transform = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)

    def __call__(self, data_dir):
        imgfiles = [f for f in os.listdir(data_dir) if f.endswith(".png")]
        imgs = np.array([np.array(self.transform(Image.open(os.path.join(data_dir, f)))) for f in imgfiles])
        with open(os.path.join(data_dir, "hrData.txt")) as hrfile:
            hrdata = np.array([float(line.split()[-1]) for line in hrfile.readlines()])
        try:
            assert len(imgs) == len(hrdata)
        except AssertionError:
            print("Not the same length!")
            return None
        length = [len(imgs)]
        imgs = torch.tensor(imgs, dtype=torch.uint8, device=device).unsqueeze_(0)
        flows = torch.tensor(self.denseflow(imgs, length), dtype=torch.float32, device=device)
        hrdata = torch.tensor(hrdata, dtype=torch.float32, device=device).unsqueeze_(0)
        return imgs, flows, hrdata, length


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True
        print("Running on GPU.")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Running on CPU.")
    print("Load model: ", end="")
    start = time.perf_counter()
    model = Model(NUM_CLASSES, NUM_FRAMES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    end = time.perf_counter()
    print("Done. Time: %.2fms." % (1000 * (end - start)))
    print("Load data: ", end="")
    start = time.perf_counter()
    loader = LoadData(NUM_FRAMES)
    images, flows, rates, length = loader(DATA_DIR)
    end = time.perf_counter()
    print("Done. Time: %.2fms." % (1000 * (end - start)))
    print("Processing: ", end="")
    start = time.perf_counter()
    if CONTINUE_UPDATE:
        model.train()
        criterion = nn.BCELoss().to(device)
        optimizer = optim.SGD(model.parameters(), 1e-5, mometum=0.9, weight_decay=3e-4)
        output = model(images, flows, rates, length)
        print("Done.\nCredibility: %.2f%%" % (output.item() * 100))
        end = time.perf_counter()
        print("Time: %.2fms" % (1000 * (end - start)))
        ans = int(input("Correct answer:\n1. Lie\n2.Truth"))
        target = torch.tensor([[ans]], dtype=torch.float32, device=device)
        loss = criterion(output, target.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            output = model(images, flows, rates, length)
            print("Done.\nCredibility: %.2f%%" % (output.item() * 100))
        end = time.perf_counter()
        print("Time: %.2fms" % (1000 * (end - start)))
    if torch.cuda.is_available(): torch.cuda.empty_cache()
