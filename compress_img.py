import os
from torchvision import transforms
from PIL import Image


def transform(shape):
    w, h = shape
    sqr_size = min(w, h)
    return transforms.Compose([
        transforms.CenterCrop(0.7 * sqr_size),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    ])


def compress_img(imgfile):
    img = Image.open(imgfile)
    imgshape = img.size
    img = transform(imgshape)(img)
    img.save(imgfile)


if __name__ == "__main__":
    for root, dirs, files in os.walk("./second_data"):
        for file in files:
            if ".DS_Store" in file:
                os.remove(os.path.join(root, file))
            elif file.endswith(".png"):
                compress_img(os.path.join(root, file))
