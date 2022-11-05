import os
import time
import torch
from torchvision import transforms
from PIL import Image

MODEL_PATH = "./fer_base_model.pth                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            .pth"
IMAGE_DIR = "./affectnet_hq/archive/anger"
CATEGORY_DICT = ("Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")

def load_image(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        # transforms.RandomCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)


if __name__ == "__main__":
    imgs = [os.path.join(IMAGE_DIR, file) for file in os.listdir(IMAGE_DIR)]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    with torch.no_grad():
        for img in imgs:
            print("Image:", img)
            start = time.perf_counter()
            src = load_image(img).unsqueeze(0).to(device)
            output = model(src)
            pred = torch.nn.functional.softmax(output.squeeze(0), 0)
            end = time.perf_counter()
            for i in range(8):
                print("%-10s: %5.2f%%" % (CATEGORY_DICT[i], pred[i] * 100))
            print("Time: %.2fms\n" % (1000 * (end - start)))
