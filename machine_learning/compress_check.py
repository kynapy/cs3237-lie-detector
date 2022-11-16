import os, glob
from torchvision import transforms
from PIL import Image

DATASETS = "./second_data"

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
    for root, dirs, files in os.walk(DATASETS):
        for file in files:
            if ".DS_Store" in file:
                os.remove(os.path.join(root, file))
            elif file.endswith(".png"):
                compress_img(os.path.join(root, file))
    
    dirs = glob.glob(os.path.join(DATASETS, "dataset_*/*/*/"))
    for dir in dirs:
        num_photos = len([f for f in os.listdir(dir) if f.endswith(".png")])
        with open(os.path.join(dir, "hrData.txt")) as f:
            num_hr = len(f.readlines())
        if num_photos != num_hr: 
            print(dir)
